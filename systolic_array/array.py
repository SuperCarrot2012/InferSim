"""2D systolic array grid with registered (R-systolic) data movement."""

from __future__ import annotations

from systolic_array.config import ArrayConfig
from systolic_array.pe import ProcessingElement
from systolic_array.types import LinkAnim, PEPhase, PESnapshot


class SystolicArray:
    """Registered systolic array: activations/psums use previous-cycle state."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        self.pes: list[list[ProcessingElement]] = [
            [ProcessingElement(r, c) for c in range(config.cols)]
            for r in range(config.rows)
        ]
        self.act_state: list[list[float | None]] = [
            [None] * config.cols for _ in range(config.rows)
        ]
        self.psum_state: list[list[float]] = [
            [0.0] * config.cols for _ in range(config.rows)
        ]
        self.weight_state: list[list[float | None]] = [
            [None] * config.cols for _ in range(config.rows)
        ]

    def reset(self) -> None:
        for row in self.pes:
            for pe in row:
                pe.reset()
        self.act_state = [[None] * self.cols for _ in range(self.rows)]
        self.psum_state = [[0.0] * self.cols for _ in range(self.rows)]
        self.weight_state = [[None] * self.cols for _ in range(self.rows)]

    def load_weights(self, k: int, n: int) -> None:
        for r in range(k):
            for c in range(n):
                pe = self.pes[r][c]
                pe.load_weight()
                pe.w_coord = [r, c]

    def begin_compute(self) -> None:
        for row in self.pes:
            for pe in row:
                if pe.weight is not None:
                    pe.begin_compute()

    def set_output_row(self, output_row: int, k: int, n: int) -> None:
        """Legacy helper; p_coord is set per-cycle in annotate_coords."""
        for r in range(k):
            for c in range(n):
                self.pes[r][c].p_coord = [output_row, c]

    def load_psums(self, m: int, n: int) -> None:
        for r in range(m):
            for c in range(n):
                self.pes[r][c].load_psum(r, c)

    def begin_compute_os(self, m: int, n: int) -> None:
        for r in range(m):
            for c in range(n):
                self.pes[r][c].begin_compute()

    def tick_os(
        self,
        left_inputs: list[float | None],
        top_inputs: list[float | None],
    ) -> tuple[list[list[float | None]], list[list[float | None]]]:
        """OS: activations flow horizontally, weights flow vertically."""
        next_act: list[list[float | None]] = [
            [None] * self.cols for _ in range(self.rows)
        ]
        next_weight: list[list[float | None]] = [
            [None] * self.cols for _ in range(self.rows)
        ]

        for r in range(self.rows):
            for c in range(self.cols):
                act_in = left_inputs[r] if c == 0 else self.act_state[r][c - 1]
                weight_in = top_inputs[c] if r == 0 else self.weight_state[r - 1][c]
                act_out, weight_out = self.pes[r][c].step_os(act_in, weight_in)
                next_act[r][c] = act_out
                next_weight[r][c] = weight_out

        self.act_state = next_act
        self.weight_state = next_weight
        return next_act, next_weight

    def annotate_coords_os(
        self, m_dim: int, k_dim: int, n_dim: int, local_cycle: int
    ) -> None:
        """Set streaming W/A and stationary P coordinates for OS dataflow."""
        for r in range(m_dim):
            for c in range(n_dim):
                pe = self.pes[r][c]
                pe.a_coord = None
                pe.w_coord = None
                pe.p_coord = [r, c]
                k_idx = local_cycle - r - c
                if pe.act_in is not None and 0 <= k_idx < k_dim:
                    pe.a_coord = [r, k_idx]
                if pe.weight_in is not None and 0 <= k_idx < k_dim:
                    pe.w_coord = [k_idx, c]
                pe.writeback = local_cycle == r + c + k_dim - 1

    def capture_pes_full(
        self,
        tile_rows: int,
        tile_cols: int,
        *,
        os_mode: bool = False,
    ) -> list[list[PESnapshot]]:
        """Capture the full array; mark PEs outside the active tile as idle."""
        result: list[list[PESnapshot]] = []
        for r in range(self.rows):
            row: list[PESnapshot] = []
            for c in range(self.cols):
                pe = self.pes[r][c]
                in_tile = r < tile_rows and c < tile_cols
                if not in_tile:
                    row.append(
                        PESnapshot(
                            row=r,
                            col=c,
                            phase=PEPhase.IDLE.value,
                            in_tile=False,
                        )
                    )
                    continue
                if os_mode:
                    row.append(
                        PESnapshot(
                            row=r,
                            col=c,
                            phase=pe.phase.value,
                            w_coord=pe.w_coord,
                            a_coord=pe.a_coord,
                            p_coord=pe.p_coord,
                            has_act=pe.act_in is not None,
                            has_weight=pe.weight_in is not None,
                            has_psum=pe.p_coord is not None,
                            writeback=pe.writeback,
                            in_tile=True,
                        )
                    )
                else:
                    row.append(
                        PESnapshot(
                            row=r,
                            col=c,
                            phase=pe.phase.value,
                            w_coord=pe.w_coord,
                            a_coord=pe.a_coord,
                            p_coord=pe.p_coord,
                            has_act=pe.act_in is not None,
                            has_weight=False,
                            has_psum=pe.psum_out != 0.0,
                            in_tile=True,
                        )
                    )
            result.append(row)
        return result

    def build_links_os(self, m: int, n: int) -> list[LinkAnim]:
        links: list[LinkAnim] = []
        for r in range(m):
            for c in range(n):
                pe = self.pes[r][c]
                if pe.act_in is not None and pe.a_coord:
                    label = f"A[{pe.a_coord[0]},{pe.a_coord[1]}]"
                    if c == 0:
                        links.append(LinkAnim(-1, -1, r, 0, label, "right"))
                    else:
                        links.append(LinkAnim(r, c - 1, r, c, label, "right"))
                if pe.weight_in is not None and pe.w_coord:
                    label = f"W[{pe.w_coord[0]},{pe.w_coord[1]}]"
                    if r == 0:
                        links.append(LinkAnim(-1, c, 0, c, label, "down"))
                    else:
                        links.append(LinkAnim(r - 1, c, r, c, label, "down"))
        return links

    def tick(
        self,
        left_inputs: list[float | None],
    ) -> tuple[list[list[float | None]], list[list[float]], list[float]]:
        next_act: list[list[float | None]] = [
            [None] * self.cols for _ in range(self.rows)
        ]
        next_psum: list[list[float]] = [
            [0.0] * self.cols for _ in range(self.rows)
        ]
        bottom_outputs: list[float] = [0.0] * self.cols

        for r in range(self.rows):
            for c in range(self.cols):
                act_in = left_inputs[r] if c == 0 else self.act_state[r][c - 1]
                psum_in = 0.0 if r == 0 else self.psum_state[r - 1][c]
                act_out, psum_out = self.pes[r][c].step(act_in, psum_in)
                next_act[r][c] = act_out
                next_psum[r][c] = psum_out
                if r == self.rows - 1:
                    bottom_outputs[c] = psum_out

        self.act_state = next_act
        self.psum_state = next_psum
        return next_act, next_psum, bottom_outputs

    def annotate_coords(
        self, m_dim: int, k: int, n: int, local_cycle: int
    ) -> None:
        """Set A[m,k] and P[m,n] coordinates on PEs for the current cycle."""
        for r in range(k):
            for c in range(n):
                pe = self.pes[r][c]
                pe.a_coord = None
                m_idx = local_cycle - r - c
                if pe.act_in is not None and 0 <= m_idx < m_dim:
                    pe.a_coord = [m_idx, r]
                if 0 <= m_idx < m_dim:
                    pe.p_coord = [m_idx, c]

    def capture_pes(self, tile_k: int, tile_n: int) -> list[list[PESnapshot]]:
        return self.capture_pes_full(tile_k, tile_n, os_mode=False)

    def build_links(self, k: int, n: int, _left_labels: list[str | None]) -> list[LinkAnim]:
        links: list[LinkAnim] = []
        for r in range(k):
            for c in range(n):
                pe = self.pes[r][c]
                if pe.act_in is not None and pe.a_coord:
                    label = f"A[{pe.a_coord[0]},{pe.a_coord[1]}]"
                    if c == 0:
                        links.append(LinkAnim(-1, -1, r, 0, label, "right"))
                    else:
                        links.append(LinkAnim(r, c - 1, r, c, label, "right"))
                if r > 0 and pe.psum_in != 0.0 and pe.p_coord:
                    label = f"P[{pe.p_coord[0]},{pe.p_coord[1]}]"
                    links.append(LinkAnim(r - 1, c, r, c, label, "down"))
        return links

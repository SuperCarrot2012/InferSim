"""2D systolic array grid with registered (R-systolic) data movement."""

from __future__ import annotations

from systolic_array_model.config import ArrayConfig
from systolic_array_model.mac import MacUnit
from systolic_array_model.types import LinkAnim, MacPhase, MacSnapshot


class SystolicArray:
    """Registered systolic array: activations/psums use previous-cycle state."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        self.macs: list[list[MacUnit]] = [
            [MacUnit(r, c) for c in range(config.cols)]
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
        for row in self.macs:
            for mac in row:
                mac.reset()
        self.act_state = [[None] * self.cols for _ in range(self.rows)]
        self.psum_state = [[0.0] * self.cols for _ in range(self.rows)]
        self.weight_state = [[None] * self.cols for _ in range(self.rows)]

    def load_weights(self, k: int, n: int, *, k0: int = 0, n0: int = 0) -> None:
        for r in range(k):
            for c in range(n):
                mac = self.macs[r][c]
                mac.load_weight()
                mac.w_coord = [r + k0, c + n0]

    def begin_compute(self) -> None:
        for row in self.macs:
            for mac in row:
                if mac.weight is not None:
                    mac.begin_compute()

    def set_output_row(self, output_row: int, k: int, n: int) -> None:
        """Legacy helper; p_coord is set per-cycle in annotate_coords."""
        for r in range(k):
            for c in range(n):
                self.macs[r][c].p_coord = [output_row, c]

    def load_psums(self, m: int, n: int, *, m0: int = 0, n0: int = 0) -> None:
        for r in range(m):
            for c in range(n):
                self.macs[r][c].load_psum(r + m0, c + n0)

    def begin_compute_os(self, m: int, n: int) -> None:
        for r in range(m):
            for c in range(n):
                self.macs[r][c].begin_compute()

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
                act_out, weight_out = self.macs[r][c].step_os(act_in, weight_in)
                next_act[r][c] = act_out
                next_weight[r][c] = weight_out

        self.act_state = next_act
        self.weight_state = next_weight
        return next_act, next_weight

    def annotate_coords_os(
        self,
        m_dim: int,
        k_dim: int,
        n_dim: int,
        local_cycle: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> None:
        """Set streaming W/A and stationary P coordinates for OS dataflow."""
        for r in range(m_dim):
            for c in range(n_dim):
                mac = self.macs[r][c]
                mac.a_coord = None
                mac.w_coord = None
                mac.p_coord = [r + m0, c + n0]
                k_idx = local_cycle - r - c
                if mac.act_in is not None and 0 <= k_idx < k_dim:
                    mac.a_coord = [r + m0, k_idx + k0]
                if mac.weight_in is not None and 0 <= k_idx < k_dim:
                    mac.w_coord = [k_idx + k0, c + n0]
                mac.writeback = local_cycle == r + c + k_dim

    def capture_macs_full(
        self,
        tile_rows: int,
        tile_cols: int,
        *,
        os_mode: bool = False,
    ) -> list[list[MacSnapshot]]:
        """Capture the full array; mark MAC units outside the active tile as idle."""
        result: list[list[MacSnapshot]] = []
        for r in range(self.rows):
            row: list[MacSnapshot] = []
            for c in range(self.cols):
                mac = self.macs[r][c]
                in_tile = r < tile_rows and c < tile_cols
                if not in_tile:
                    row.append(
                        MacSnapshot(
                            row=r,
                            col=c,
                            phase=MacPhase.IDLE.value,
                            in_tile=False,
                        )
                    )
                    continue
                if os_mode:
                    row.append(
                        MacSnapshot(
                            row=r,
                            col=c,
                            phase=mac.phase.value,
                            w_coord=mac.w_coord,
                            a_coord=mac.a_coord,
                            p_coord=mac.p_coord,
                            has_act=mac.act_in is not None,
                            has_weight=mac.weight_in is not None,
                            has_psum=mac.p_coord is not None,
                            writeback=mac.writeback,
                            in_tile=True,
                        )
                    )
                else:
                    row.append(
                        MacSnapshot(
                            row=r,
                            col=c,
                            phase=mac.phase.value,
                            w_coord=mac.w_coord,
                            a_coord=mac.a_coord,
                            p_coord=mac.p_coord,
                            has_act=mac.act_in is not None,
                            has_weight=False,
                            has_psum=mac.psum_out != 0.0,
                            in_tile=True,
                        )
                    )
            result.append(row)
        return result

    def build_links_os(self, m: int, n: int) -> list[LinkAnim]:
        links: list[LinkAnim] = []
        for r in range(m):
            for c in range(n):
                mac = self.macs[r][c]
                if mac.act_in is not None and mac.a_coord:
                    label = f"A[{mac.a_coord[0]},{mac.a_coord[1]}]"
                    if c == 0:
                        links.append(LinkAnim(-1, -1, r, 0, label, "right"))
                    else:
                        links.append(LinkAnim(r, c - 1, r, c, label, "right"))
                if mac.weight_in is not None and mac.w_coord:
                    label = f"W[{mac.w_coord[0]},{mac.w_coord[1]}]"
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
                act_out, psum_out = self.macs[r][c].step(act_in, psum_in)
                next_act[r][c] = act_out
                next_psum[r][c] = psum_out
                if r == self.rows - 1:
                    bottom_outputs[c] = psum_out

        self.act_state = next_act
        self.psum_state = next_psum
        return next_act, next_psum, bottom_outputs

    def annotate_coords(
        self,
        m_dim: int,
        k: int,
        n: int,
        local_cycle: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> None:
        """Set A[m,k] and P[m,n] coordinates on MAC units for the current cycle."""
        for r in range(k):
            for c in range(n):
                mac = self.macs[r][c]
                mac.a_coord = None
                m_idx = local_cycle - r - c
                if mac.act_in is not None and 0 <= m_idx < m_dim:
                    mac.a_coord = [m_idx + m0, r + k0]
                if 0 <= m_idx < m_dim:
                    mac.p_coord = [m_idx + m0, c + n0]

    def capture_macs(self, tile_k: int, tile_n: int) -> list[list[MacSnapshot]]:
        return self.capture_macs_full(tile_k, tile_n, os_mode=False)

    def build_links(self, k: int, n: int, _left_labels: list[str | None]) -> list[LinkAnim]:
        links: list[LinkAnim] = []
        for r in range(k):
            for c in range(n):
                mac = self.macs[r][c]
                if mac.act_in is not None and mac.a_coord:
                    label = f"A[{mac.a_coord[0]},{mac.a_coord[1]}]"
                    if c == 0:
                        links.append(LinkAnim(-1, -1, r, 0, label, "right"))
                    else:
                        links.append(LinkAnim(r, c - 1, r, c, label, "right"))
                if r > 0 and mac.psum_in != 0.0 and mac.p_coord:
                    label = f"P[{mac.p_coord[0]},{mac.p_coord[1]}]"
                    links.append(LinkAnim(r - 1, c, r, c, label, "down"))
        return links

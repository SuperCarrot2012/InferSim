"""Weight-stationary dataflow for C = A @ B (TPU-style, coordinate-only visualization)."""

from __future__ import annotations

from systolic_array_model.array import SystolicArray
from systolic_array_model.config import ArrayConfig
from systolic_array_model.dataflow.base import DataflowStrategy
from systolic_array_model.memory import compute_memory_access
from systolic_array_model.cycles import compute_cycle_count
from systolic_array_model.types import CycleSnapshot, DataflowType, SimPhase


class WeightStationaryDataflow(DataflowStrategy):
    """WS GEMM: MAC(r,c) holds W[r,c]; A streams with diagonal wavefront."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config

    def run(
        self,
        array: SystolicArray,
        m: int,
        k: int,
        n: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> list[CycleSnapshot]:
        rows, cols = array.rows, array.cols

        if k > rows or n > cols:
            raise ValueError(
                f"K and N must fit in array ({rows}x{cols}); got K={k}, N={n}"
            )

        snapshots: list[CycleSnapshot] = []
        cycle = 0

        array.load_weights(k, n, k0=k0, n0=n0)
        array.begin_compute()
        array.act_state = [[None] * cols for _ in range(rows)]
        array.psum_state = [[0.0] * cols for _ in range(rows)]

        # Single continuous GEMM: total compute cycles = M + K + N - 2.
        # At cycle t, array row r injects A[t-r, r] when 0 <= t-r < M.
        max_local = max(0, m + k + n - 3)

        for local_cycle in range(max_local + 1):
            left_inputs: list[float | None] = [None] * rows
            left_labels: list[str | None] = [None] * rows
            for r in range(k):
                m_idx = local_cycle - r
                if 0 <= m_idx < m:
                    left_inputs[r] = 1.0
                    left_labels[r] = f"A[{m_idx + m0},{r + k0}]"

            array.tick(left_inputs)
            array.annotate_coords(m, k, n, local_cycle, m0=m0, k0=k0, n0=n0)

            bottom_labels: list[str | None] = [None] * n
            for c in range(n):
                mac = array.macs[k - 1][c]
                if mac.psum_out != 0.0 and mac.p_coord:
                    bottom_labels[c] = f"P[{mac.p_coord[0]},{mac.p_coord[1]}]"

            snapshots.append(
                self._make_snapshot(
                    array,
                    cycle,
                    SimPhase.COMPUTE,
                    left_labels,
                    m,
                    k,
                    n,
                    local_cycle=local_cycle,
                    max_local=max_local,
                    bottom_labels=bottom_labels,
                )
            )
            cycle += 1

        return snapshots

    def snapshot_at_cycle(
        self,
        array: SystolicArray,
        m: int,
        k: int,
        n: int,
        cycle: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> CycleSnapshot:
        rows, cols = array.rows, array.cols
        if k > rows or n > cols:
            raise ValueError(
                f"K and N must fit in array ({rows}x{cols}); got K={k}, N={n}"
            )

        total = compute_cycle_count(m, k, n)
        if cycle < 0 or cycle >= total:
            raise ValueError(f"Cycle {cycle} out of range [0, {total})")

        array.reset()
        array.load_weights(k, n, k0=k0, n0=n0)
        array.begin_compute()
        array.act_state = [[None] * cols for _ in range(rows)]
        array.psum_state = [[0.0] * cols for _ in range(rows)]

        max_local = max(0, m + k + n - 3)
        for local_cycle in range(max_local + 1):
            left_inputs: list[float | None] = [None] * rows
            left_labels: list[str | None] = [None] * rows
            for r in range(k):
                m_idx = local_cycle - r
                if 0 <= m_idx < m:
                    left_inputs[r] = 1.0
                    left_labels[r] = f"A[{m_idx + m0},{r + k0}]"

            array.tick(left_inputs)
            array.annotate_coords(m, k, n, local_cycle, m0=m0, k0=k0, n0=n0)

            if local_cycle == cycle:
                bottom_labels: list[str | None] = [None] * n
                for c in range(n):
                    mac = array.macs[k - 1][c]
                    if mac.psum_out != 0.0 and mac.p_coord:
                        bottom_labels[c] = f"P[{mac.p_coord[0]},{mac.p_coord[1]}]"

                return self._make_snapshot(
                    array,
                    cycle,
                    SimPhase.COMPUTE,
                    left_labels,
                    m,
                    k,
                    n,
                    local_cycle=local_cycle,
                    max_local=max_local,
                    bottom_labels=bottom_labels,
                )

        raise RuntimeError(f"Failed to capture WS snapshot at cycle {cycle}")

    def _make_snapshot(
        self,
        array: SystolicArray,
        cycle: int,
        phase: SimPhase,
        left_labels: list[str | None],
        m: int,
        k: int,
        n: int,
        local_cycle: int,
        max_local: int,
        bottom_labels: list[str | None] | None = None,
    ) -> CycleSnapshot:
        left_padded = left_labels[: array.rows]
        bottom_padded = (bottom_labels + [None] * array.cols)[: array.cols]

        memory = compute_memory_access(
            phase=phase,
            dataflow=DataflowType.WEIGHT_STATIONARY,
            m=m,
            k=k,
            n=n,
            left_labels=left_padded[:k],
            bottom_labels=bottom_padded[:n],
        )

        return CycleSnapshot(
            cycle=cycle,
            phase=phase.value,
            macs=array.capture_macs(k, n),
            left_inject=left_padded,
            top_inject=[],
            bottom_output=bottom_padded,
            active_links=array.build_links(k, n, left_labels),
            progress={
                "m": m,
                "k": k,
                "n": n,
                "local_cycle": local_cycle,
                "total_local_cycles": max_local,
                "compute_cycles": m + k + n - 2,
                "active_rows": k,
                "active_cols": n,
                "dataflow": DataflowType.WEIGHT_STATIONARY.value,
            },
            memory=memory,
        )

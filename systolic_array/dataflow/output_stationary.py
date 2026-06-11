"""Output (partial-sum) stationary dataflow for C = A @ B."""

from __future__ import annotations

from systolic_array.array import SystolicArray
from systolic_array.config import ArrayConfig
from systolic_array.dataflow.base import DataflowStrategy
from systolic_array.memory import compute_memory_access
from systolic_array.cycles import compute_cycle_count
from systolic_array.types import CycleSnapshot, DataflowType, SimPhase


class OutputStationaryDataflow(DataflowStrategy):
    """OS GEMM: PE(m,c) holds P[m,c]; A streams right, W streams down."""

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

        if m > rows or n > cols:
            raise ValueError(
                f"M and N must fit in array ({rows}x{cols}); got M={m}, N={n}"
            )

        snapshots: list[CycleSnapshot] = []
        cycle = 0

        array.load_psums(m, n, m0=m0, n0=n0)
        array.begin_compute_os(m, n)
        array.act_state = [[None] * cols for _ in range(rows)]
        array.weight_state = [[None] * cols for _ in range(rows)]

        max_local = max(0, m + k + n - 3)

        for local_cycle in range(max_local + 1):
            left_inputs: list[float | None] = [None] * rows
            left_labels: list[str | None] = [None] * rows
            top_inputs: list[float | None] = [None] * cols
            top_labels: list[str | None] = [None] * cols

            for r in range(m):
                k_idx = local_cycle - r
                if 0 <= k_idx < k:
                    left_inputs[r] = 1.0
                    left_labels[r] = f"A[{r + m0},{k_idx + k0}]"

            for c in range(n):
                k_idx = local_cycle - c
                if 0 <= k_idx < k:
                    top_inputs[c] = 1.0
                    top_labels[c] = f"W[{k_idx + k0},{c + n0}]"

            array.tick_os(left_inputs, top_inputs)
            array.annotate_coords_os(m, k, n, local_cycle, m0=m0, k0=k0, n0=n0)

            snapshots.append(
                self._make_snapshot(
                    array,
                    cycle,
                    SimPhase.COMPUTE,
                    left_labels,
                    top_labels,
                    m,
                    k,
                    n,
                    local_cycle=local_cycle,
                    max_local=max_local,
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
        if m > rows or n > cols:
            raise ValueError(
                f"M and N must fit in array ({rows}x{cols}); got M={m}, N={n}"
            )

        total = compute_cycle_count(m, k, n)
        if cycle < 0 or cycle >= total:
            raise ValueError(f"Cycle {cycle} out of range [0, {total})")

        array.reset()
        array.load_psums(m, n, m0=m0, n0=n0)
        array.begin_compute_os(m, n)
        array.act_state = [[None] * cols for _ in range(rows)]
        array.weight_state = [[None] * cols for _ in range(rows)]

        max_local = max(0, m + k + n - 3)
        for local_cycle in range(max_local + 1):
            left_inputs: list[float | None] = [None] * rows
            left_labels: list[str | None] = [None] * rows
            top_inputs: list[float | None] = [None] * cols
            top_labels: list[str | None] = [None] * cols

            for r in range(m):
                k_idx = local_cycle - r
                if 0 <= k_idx < k:
                    left_inputs[r] = 1.0
                    left_labels[r] = f"A[{r + m0},{k_idx + k0}]"

            for c in range(n):
                k_idx = local_cycle - c
                if 0 <= k_idx < k:
                    top_inputs[c] = 1.0
                    top_labels[c] = f"W[{k_idx + k0},{c + n0}]"

            array.tick_os(left_inputs, top_inputs)
            array.annotate_coords_os(m, k, n, local_cycle, m0=m0, k0=k0, n0=n0)

            if local_cycle == cycle:
                return self._make_snapshot(
                    array,
                    cycle,
                    SimPhase.COMPUTE,
                    left_labels,
                    top_labels,
                    m,
                    k,
                    n,
                    local_cycle=local_cycle,
                    max_local=max_local,
                )

        raise RuntimeError(f"Failed to capture OS snapshot at cycle {cycle}")

    def snapshot_after_drain(
        self,
        array: SystolicArray,
        m: int,
        k: int,
        n: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> CycleSnapshot:
        """One drain tick after compute ends: W/A leave PEs, only P coords remain."""
        rows, cols = array.rows, array.cols
        if m > rows or n > cols:
            raise ValueError(
                f"M and N must fit in array ({rows}x{cols}); got M={m}, N={n}"
            )

        total = compute_cycle_count(m, k, n)
        array.reset()
        array.load_psums(m, n, m0=m0, n0=n0)
        array.begin_compute_os(m, n)
        array.act_state = [[None] * cols for _ in range(rows)]
        array.weight_state = [[None] * cols for _ in range(rows)]

        max_local = max(0, m + k + n - 3)
        for local_cycle in range(max_local + 1):
            left_inputs: list[float | None] = [None] * rows
            top_inputs: list[float | None] = [None] * cols

            for r in range(m):
                k_idx = local_cycle - r
                if 0 <= k_idx < k:
                    left_inputs[r] = 1.0

            for c in range(n):
                k_idx = local_cycle - c
                if 0 <= k_idx < k:
                    top_inputs[c] = 1.0

            array.tick_os(left_inputs, top_inputs)
            array.annotate_coords_os(m, k, n, local_cycle, m0=m0, k0=k0, n0=n0)

        drain_cycle = max_local + 1
        array.tick_os([None] * rows, [None] * cols)
        array.annotate_coords_os(m, k, n, drain_cycle, m0=m0, k0=k0, n0=n0)

        return self._make_snapshot(
            array,
            total,
            SimPhase.DONE,
            [None] * rows,
            [None] * cols,
            m,
            k,
            n,
            local_cycle=drain_cycle,
            max_local=max_local,
        )

    def _make_snapshot(
        self,
        array: SystolicArray,
        cycle: int,
        phase: SimPhase,
        left_labels: list[str | None],
        top_labels: list[str | None],
        m: int,
        k: int,
        n: int,
        local_cycle: int,
        max_local: int,
    ) -> CycleSnapshot:
        left_padded = left_labels[: array.rows]
        top_padded = top_labels[: array.cols]

        memory = compute_memory_access(
            phase=phase,
            dataflow=DataflowType.OUTPUT_STATIONARY,
            m=m,
            k=k,
            n=n,
            left_labels=left_padded[:m],
            top_labels=top_padded[:n],
            local_cycle=local_cycle,
        )

        return CycleSnapshot(
            cycle=cycle,
            phase=phase.value,
            pes=array.capture_pes_full(m, n, os_mode=True),
            left_inject=left_padded,
            top_inject=top_padded,
            bottom_output=[None] * array.cols,
            active_links=array.build_links_os(m, n),
            progress={
                "m": m,
                "k": k,
                "n": n,
                "local_cycle": local_cycle,
                "total_local_cycles": max_local,
                "compute_cycles": m + k + n - 2,
                "active_rows": m,
                "active_cols": n,
                "dataflow": DataflowType.OUTPUT_STATIONARY.value,
            },
            memory=memory,
        )

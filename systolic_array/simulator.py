"""Public entry point for systolic array simulation."""

from __future__ import annotations

from typing import Any

from systolic_array.config import ArrayConfig
from systolic_array.engine import CycleEngine
from systolic_array.types import DataflowType, SimResult


def run_simulation(
    m: int,
    k: int,
    n: int,
    *,
    rows: int = 16,
    cols: int = 16,
    dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY,
    mac_latency: int = 1,
    weight_load_cycles: int = 1,
) -> SimResult:
    """Run cycle-accurate WS GEMM dataflow simulation.

    Only models data movement; no numeric computation or result verification.
    Constraints: K <= rows, N <= cols.
    """
    if mac_latency != 1:
        raise NotImplementedError("mac_latency > 1 planned for Phase 2")

    config = ArrayConfig(
        rows=rows,
        cols=cols,
        dataflow=dataflow,
        mac_latency=mac_latency,
        weight_load_cycles=weight_load_cycles,
    )
    engine = CycleEngine(config)
    return engine.run(m, k, n)


def result_to_dict(result: SimResult) -> dict[str, Any]:
    return result.to_dict()

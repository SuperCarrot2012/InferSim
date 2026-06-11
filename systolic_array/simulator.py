"""Public entry point for systolic array simulation."""

from __future__ import annotations

from typing import Any

from systolic_array.config import ArrayConfig
from systolic_array.cycles import compute_cycle_count
from systolic_array.engine import CycleEngine
from systolic_array.logic_die_engine import LogicDieEngine
from systolic_array.memory import empty_memory_access, merge_memory_access
from systolic_array.tensor_core_engine import TensorCoreEngine
from systolic_array.types import CycleSnapshot, DataflowType, SimResult


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
    """Run cycle-accurate GEMM dataflow simulation.

    OS: Logic Die (48 PPU) → PPU (4×4 core) → Core (16×16 PE); M≤16, N split first.
    WS: unchanged 4×4 Tensor Core tiling.
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
    if dataflow == DataflowType.OUTPUT_STATIONARY:
        engine = LogicDieEngine(config)
    else:
        engine = TensorCoreEngine(config)
    return engine.run(m, k, n)


def result_to_dict(result: SimResult) -> dict[str, Any]:
    return result.to_dict()


def _tile_key_from_dict(tile: dict[str, Any]) -> str:
    if "ppu_index" in tile:
        return f"{tile['ppu_index']},{tile['core_row']},{tile['core_col']}"
    return f"{tile['grid_row']},{tile['grid_col']}"


def _find_tile(tile_plan: dict[str, Any], tile_key: str) -> dict[str, Any]:
    for tile in tile_plan.get("tiles", []):
        if _tile_key_from_dict(tile) == tile_key:
            return tile
    raise KeyError(tile_key)


def render_micro_snapshot(
    result: SimResult,
    tile_key: str,
    cycle: int,
) -> CycleSnapshot:
    """Re-run one tile up to ``cycle`` and capture its Systolic Core View snapshot."""
    tile = _find_tile(result.tile_plan, tile_key)
    config = ArrayConfig.from_dict(result.config)
    total = compute_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
    if total <= 0:
        raise IndexError(f"Tile {tile_key} has no compute cycles")
    if cycle < 0:
        raise IndexError(f"Cycle {cycle} out of range [0, {total})")

    engine = CycleEngine(config)
    if cycle >= total:
        # Post-compute drain: W/A have left PEs; only P coords remain.
        return engine.snapshot_after_drain(
            tile["local_m"],
            tile["local_k"],
            tile["local_n"],
            m0=tile["m0"],
            k0=tile["k0"],
            n0=tile["n0"],
        )

    return engine.snapshot_at_cycle(
        tile["local_m"],
        tile["local_k"],
        tile["local_n"],
        cycle,
        m0=tile["m0"],
        k0=tile["k0"],
        n0=tile["n0"],
    )


def aggregate_ppu_memory(
    result: SimResult,
    ppu_index: int,
    cycle: int,
) -> dict[str, object]:
    """Sum per-cycle memory traffic across all active systolic cores in one PPU."""
    tiles = [
        t
        for t in result.tile_plan.get("tiles", [])
        if t.get("ppu_index") == ppu_index
    ]
    if not tiles:
        raise KeyError(f"PPU {ppu_index} has no tiles")

    total: dict[str, int | str | dict[str, int]] | None = None
    contributing = 0

    for tile in tiles:
        tile_cycles = compute_cycle_count(
            tile["local_m"], tile["local_k"], tile["local_n"]
        )
        if cycle >= tile_cycles:
            continue

        tile_key = _tile_key_from_dict(tile)
        snap = render_micro_snapshot(result, tile_key, cycle)
        mem = snap.memory or empty_memory_access()
        total = mem if total is None else merge_memory_access(total, mem)
        contributing += 1

    if total is None:
        total = empty_memory_access()

    return {**total, "contributing_cores": contributing}

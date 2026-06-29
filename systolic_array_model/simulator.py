"""Public entry point for systolic array simulation."""

from __future__ import annotations

from typing import Any

from systolic_array_model.config import ArrayConfig
from systolic_array_model.cycles import compute_cycle_count
from systolic_array_model.engine import CycleEngine
from systolic_array_model.logic_die_engine import LogicDieEngine
from systolic_array_model.hierarchy import PPU_CORE_GRID
from systolic_array_model.memory import empty_memory_access, merge_memory_access, os_memory_at_cycle, peak_memory_access
from systolic_array_model.tensor_core_engine import TensorCoreEngine
from systolic_array_model.types import CycleSnapshot, DataflowType, MacroArrayState, SimResult


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

    OS: Logic Die (32 PPU) → PPU (4×4 core) → Core (16×16 PE); M≤16, N split first.
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
        wave = tile.get("wave_index", 0)
        return f"{wave},{tile['ppu_index']},{tile['core_row']},{tile['core_col']}"
    return f"{tile['grid_row']},{tile['grid_col']}"


def _wave_local_cycle(tile_plan: dict[str, Any], global_cycle: int) -> tuple[int, int]:
    """Return (wave_index, local_cycle) for a global timeline cycle."""
    waves = tile_plan.get("waves") or []
    if not waves:
        return 0, global_cycle
    for wave in waves:
        start = int(wave["cycle_start"])
        end = int(wave["cycle_end"])
        if start <= global_cycle < end:
            return int(wave["wave_index"]), global_cycle - start
    last = waves[-1]
    wave_idx = int(last["wave_index"])
    local = max(0, global_cycle - int(last["cycle_start"]))
    end = int(last["cycle_end"])
    if global_cycle >= end:
        local = max(0, end - 1 - int(last["cycle_start"]))
    return wave_idx, local


def _tiles_for_wave(tile_plan: dict[str, Any], wave_index: int) -> list[dict[str, Any]]:
    return [
        t
        for t in tile_plan.get("tiles", [])
        if int(t.get("wave_index", 0)) == wave_index
    ]


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
    wave_idx, local_cycle = _wave_local_cycle(result.tile_plan, cycle)
    if int(tile.get("wave_index", 0)) != wave_idx:
        local_cycle = 0

    total = compute_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
    if total <= 0:
        raise IndexError(f"Tile {tile_key} has no compute cycles")
    if local_cycle < 0:
        raise IndexError(f"Cycle {local_cycle} out of range [0, {total})")

    engine = CycleEngine(config)
    if local_cycle >= total:
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
        local_cycle,
        m0=tile["m0"],
        k0=tile["k0"],
        n0=tile["n0"],
    )


def _os_core_label(tile: dict[str, Any]) -> str:
    return (
        f"M[{tile['m0']}:{tile['m0'] + tile['local_m']}]"
        f"×N[{tile['n0']}:{tile['n0'] + tile['local_n']}]"
    )


def _tile_memory_at_cycle(result: SimResult, tile: dict[str, Any], cycle: int) -> dict[str, object]:
    dataflow = DataflowType(result.config.get("dataflow", DataflowType.OUTPUT_STATIONARY.value))
    if dataflow == DataflowType.OUTPUT_STATIONARY:
        return os_memory_at_cycle(
            tile["local_m"],
            tile["local_k"],
            tile["local_n"],
            cycle,
            m0=tile["m0"],
            k0=tile["k0"],
            n0=tile["n0"],
        )
    tile_key = _tile_key_from_dict(tile)
    snap = render_micro_snapshot(result, tile_key, cycle)
    return snap.memory or empty_memory_access()


def render_ppu_core_grid(
    result: SimResult,
    ppu_index: int,
    cycle: int,
) -> list[list[dict[str, object]]]:
    """Synthesize one PPU's 4×4 core grid at a global cycle (no bulk snapshot storage)."""
    wave_idx, local_cycle = _wave_local_cycle(result.tile_plan, cycle)
    grid: list[list[dict[str, object]]] = [
        [
            MacroArrayState(active=False, grid_row=r, grid_col=c).to_dict()
            for c in range(PPU_CORE_GRID)
        ]
        for r in range(PPU_CORE_GRID)
    ]

    for tile in _tiles_for_wave(result.tile_plan, wave_idx):
        if tile.get("ppu_index") != ppu_index:
            continue
        cr, cc = tile["core_row"], tile["core_col"]
        tile_cycles = compute_cycle_count(
            tile["local_m"], tile["local_k"], tile["local_n"]
        )
        computing = local_cycle < tile_cycles
        grid[cr][cc] = MacroArrayState(
            active=True,
            computing=computing,
            done=not computing,
            grid_row=cr,
            grid_col=cc,
            local_m=tile["local_m"],
            local_k=tile["local_k"],
            local_n=tile["local_n"],
            m0=tile["m0"],
            k0=tile["k0"],
            n0=tile["n0"],
            label=_os_core_label(tile),
        ).to_dict()

    return grid


def aggregate_ppu_memory(
    result: SimResult,
    ppu_index: int,
    cycle: int,
) -> dict[str, object]:
    """Sum per-cycle memory traffic across all active systolic cores in one PPU."""
    wave_idx, local_cycle = _wave_local_cycle(result.tile_plan, cycle)
    tiles = [
        t
        for t in _tiles_for_wave(result.tile_plan, wave_idx)
        if t.get("ppu_index") == ppu_index
    ]
    if not tiles:
        raise KeyError(f"PPU {ppu_index} has no tiles in wave {wave_idx}")

    total: dict[str, int | str | dict[str, int]] | None = None
    contributing = 0

    for tile in tiles:
        tile_cycles = compute_cycle_count(
            tile["local_m"], tile["local_k"], tile["local_n"]
        )
        if local_cycle >= tile_cycles:
            continue

        mem = _tile_memory_at_cycle(result, tile, local_cycle)
        total = mem if total is None else merge_memory_access(total, mem)
        contributing += 1

    if total is None:
        total = empty_memory_access()

    return {**total, "contributing_cores": contributing}


def peak_core_memory(result: SimResult, tile_key: str) -> dict[str, object]:
    """Peak per-cycle memory bandwidth for one systolic core tile."""
    tile = _find_tile(result.tile_plan, tile_key)
    total = compute_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
    memories: list[dict[str, object]] = []
    for cycle in range(total):
        memories.append(_tile_memory_at_cycle(result, tile, cycle))
    return peak_memory_access(memories)


def peak_ppu_memory(result: SimResult, ppu_index: int) -> dict[str, object]:
    """Peak per-cycle aggregated memory bandwidth across one PPU."""
    memories: list[dict[str, object]] = []
    for wave in result.tile_plan.get("waves") or [{"cycle_start": 0, "cycle_end": result.total_cycles}]:
        start = int(wave["cycle_start"])
        end = int(wave["cycle_end"])
        for cycle in range(start, end):
            mem = aggregate_ppu_memory(result, ppu_index, cycle)
            memories.append(mem)
    return peak_memory_access(memories)

"""Public entry point for systolic array simulation."""

from __future__ import annotations

from typing import Any

from systolic_array_model.config import ArrayConfig
from systolic_array_model.cycles import compute_cycle_count, compute_os_cycle_count
from systolic_array_model.engine import CycleEngine
from systolic_array_model.logic_die_engine import LogicDieEngine
from systolic_array_model.hierarchy import PPU_CORE_GRID
from systolic_array_model.memory import (
    DEFAULT_DTYPE,
    FP16_BYTES,
    empty_memory_access,
    merge_memory_access,
    os_memory_at_cycle,
    peak_memory_access,
)
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

    OS: Logic Die (32 PPU) → PPU (4×4 core) → Core (16×16 MAC); M≤16, N split first.
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
    """Re-run one tile up to ``cycle`` and capture its Core View snapshot."""
    tile = _find_tile(result.tile_plan, tile_key)
    config = ArrayConfig.from_dict(result.config)
    wave_idx, local_cycle = _wave_local_cycle(result.tile_plan, cycle)
    if int(tile.get("wave_index", 0)) != wave_idx:
        local_cycle = 0

    total = compute_os_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
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
        tile_cycles = compute_os_cycle_count(
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


def _ppu_tiles_in_wave(
    tile_plan: dict[str, Any],
    ppu_index: int,
    wave_index: int,
) -> list[dict[str, Any]]:
    return [
        t
        for t in _tiles_for_wave(tile_plan, wave_index)
        if int(t.get("ppu_index", -1)) == ppu_index
    ]


def _scale_memory_access(
    mem: dict[str, int | str | dict[str, int]],
    factor: int,
) -> dict[str, int | str | dict[str, int]]:
    if factor <= 1:
        return mem
    rb = mem.get("read_breakdown", {})
    wb = mem.get("write_breakdown", {})
    if not isinstance(rb, dict):
        rb = {}
    if not isinstance(wb, dict):
        wb = {}
    return {
        "dtype": str(mem.get("dtype", DEFAULT_DTYPE)),
        "bytes_per_elem": int(mem.get("bytes_per_elem", FP16_BYTES)),
        "read_bytes": int(mem.get("read_bytes", 0)) * factor,
        "write_bytes": int(mem.get("write_bytes", 0)) * factor,
        "read_elems": int(mem.get("read_elems", 0)) * factor,
        "write_elems": int(mem.get("write_elems", 0)) * factor,
        "read_breakdown": {
            "weight": int(rb.get("weight", 0)) * factor,
            "activation": int(rb.get("activation", 0)) * factor,
        },
        "write_breakdown": {
            "output": int(wb.get("output", 0)) * factor,
        },
    }


def _collect_ppu_wave_memories(
    result: SimResult,
    ppu_index: int,
    wave_index: int,
) -> list[dict[str, int | str | dict[str, int]]]:
    """Per-cycle aggregated memory for one PPU within a single wave (wave-local cycles)."""
    tiles = _ppu_tiles_in_wave(result.tile_plan, ppu_index, wave_index)
    if not tiles:
        return []

    tile_cycles = [
        compute_os_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
        for tile in tiles
    ]
    max_cycles = max(tile_cycles)
    uniform_dims = len({(t["local_m"], t["local_k"], t["local_n"]) for t in tiles}) == 1

    if uniform_dims:
        rep = tiles[0]
        core_count = len(tiles)
        memories: list[dict[str, int | str | dict[str, int]]] = []
        for local_cycle in range(tile_cycles[0]):
            mem = _tile_memory_at_cycle(result, rep, local_cycle)
            if core_count > 1:
                mem = _scale_memory_access(mem, core_count)
            memories.append(mem)
        return memories

    memories = []
    for local_cycle in range(max_cycles):
        total: dict[str, int | str | dict[str, int]] | None = None
        for tile, cycles in zip(tiles, tile_cycles, strict=True):
            if local_cycle >= cycles:
                continue
            mem = _tile_memory_at_cycle(result, tile, local_cycle)
            total = mem if total is None else merge_memory_access(total, mem)
        memories.append(total or empty_memory_access())
    return memories


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
        if int(t.get("ppu_index", -1)) == ppu_index
    ]
    if not tiles:
        raise KeyError(f"PPU {ppu_index} has no tiles in wave {wave_idx}")

    total: dict[str, int | str | dict[str, int]] | None = None
    contributing = 0

    for tile in tiles:
        tile_cycles = compute_os_cycle_count(
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
    total = compute_os_cycle_count(tile["local_m"], tile["local_k"], tile["local_n"])
    memories: list[dict[str, object]] = []
    for cycle in range(total):
        memories.append(_tile_memory_at_cycle(result, tile, cycle))
    return peak_memory_access(memories)


def peak_ppu_memory(result: SimResult, ppu_index: int) -> dict[str, object]:
    """Peak per-cycle aggregated memory bandwidth across one PPU."""
    best = peak_memory_access([])
    waves = result.tile_plan.get("waves") or [{"wave_index": 0}]
    for wave in waves:
        wave_idx = int(wave.get("wave_index", 0))
        wave_memories = _collect_ppu_wave_memories(result, ppu_index, wave_idx)
        if not wave_memories:
            continue
        wave_peak = peak_memory_access(wave_memories)
        if int(wave_peak["peak_bytes"]) > int(best["peak_bytes"]):
            best = wave_peak
    return best


def logic_die_sram_bandwidth_demand(result: SimResult) -> dict[str, object]:
    """SRAM bandwidth demand: per-PPU peak × active PPU count in the busiest wave."""
    from systolic_array_model.hierarchy import DIE_PPU_COUNT

    tiles = result.tile_plan.get("tiles", [])
    if not tiles:
        return {
            "per_ppu_peak_bytes": 0,
            "logic_die_peak_bytes": 0,
            "active_ppu_count": 0,
            "die_ppu_count": DIE_PPU_COUNT,
        }

    waves = result.tile_plan.get("waves") or [{"wave_index": 0}]
    wave_stats: list[dict[str, int]] = []
    logic_die_peak = 0
    peak_per_ppu = 0
    peak_active_ppus = 0

    for wave in waves:
        wave_idx = int(wave.get("wave_index", 0))
        wave_tiles = _tiles_for_wave(result.tile_plan, wave_idx)
        ppu_indices = sorted(
            {int(t["ppu_index"]) for t in wave_tiles if t.get("ppu_index") is not None}
        )
        if not ppu_indices:
            continue

        wave_ppu_peak = 0
        for ppu_index in ppu_indices:
            wave_memories = _collect_ppu_wave_memories(result, ppu_index, wave_idx)
            peak_bytes = int(peak_memory_access(wave_memories)["peak_bytes"])
            wave_ppu_peak = max(wave_ppu_peak, peak_bytes)

        active_count = len(ppu_indices)
        wave_die_peak = wave_ppu_peak * active_count
        wave_stats.append(
            {
                "wave_index": wave_idx,
                "per_ppu_peak_bytes": wave_ppu_peak,
                "active_ppu_count": active_count,
                "logic_die_peak_bytes": wave_die_peak,
            }
        )
        if wave_die_peak > logic_die_peak:
            logic_die_peak = wave_die_peak
            peak_per_ppu = wave_ppu_peak
            peak_active_ppus = active_count

    return {
        "per_ppu_peak_bytes": peak_per_ppu,
        "logic_die_peak_bytes": logic_die_peak,
        "active_ppu_count": peak_active_ppus,
        "die_ppu_count": DIE_PPU_COUNT,
        "waves": wave_stats,
    }

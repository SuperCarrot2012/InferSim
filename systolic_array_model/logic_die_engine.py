"""Logic Die engine: OS simulation across 32 PPUs × 16 cores × 16×16 PEs."""

from __future__ import annotations

from systolic_array_model.config import ArrayConfig
from systolic_array_model.cycles import compute_cycle_count
from systolic_array_model.hierarchy import (
    DIE_PPU_COLS,
    DIE_PPU_ROWS,
    PPU_CORE_GRID,
    CoreTile,
    LogicDiePlan,
    compute_os_logic_die_plan,
    os_n_wave_chunks,
)
from systolic_array_model.types import CycleSnapshot, SimResult


def _empty_die_grid() -> list[list[dict]]:
    grid: list[list[dict]] = []
    for r in range(DIE_PPU_ROWS):
        row: list[dict] = []
        for c in range(DIE_PPU_COLS):
            idx = r * DIE_PPU_COLS + c
            row.append(
                {
                    "active": False,
                    "computing": False,
                    "done": False,
                    "ppu_index": idx,
                    "ppu_row": r,
                    "ppu_col": c,
                    "active_cores": 0,
                    "label": "",
                }
            )
        grid.append(row)
    return grid


def _merge_os_snapshots(
    plan: LogicDiePlan,
    tile_cycles: dict[str, int],
    *,
    wave_index: int = 0,
    wave_count: int = 1,
    n0: int = 0,
    wave_n: int = 0,
    cycle_offset: int = 0,
) -> list[CycleSnapshot]:
    if not plan.tiles:
        return []

    max_cycles = max(tile_cycles.values())
    merged: list[CycleSnapshot] = []

    ppu_active_cores: dict[int, int] = {}
    for tile in plan.tiles:
        ppu_active_cores[tile.ppu_index] = ppu_active_cores.get(tile.ppu_index, 0) + 1

    for local_cycle in range(max_cycles):
        die_ppuss = _empty_die_grid()
        ppu_computing: dict[int, int] = {}
        any_computing = False

        for tile in plan.tiles:
            if local_cycle < tile_cycles[tile.key]:
                any_computing = True
                ppu_computing[tile.ppu_index] = ppu_computing.get(tile.ppu_index, 0) + 1

        for ppu_index, active_cores in ppu_active_cores.items():
            pr = ppu_index // DIE_PPU_COLS
            pc = ppu_index % DIE_PPU_COLS
            computing_cores = ppu_computing.get(ppu_index, 0)
            die_ppuss[pr][pc] = {
                "active": True,
                "computing": computing_cores > 0,
                "done": active_cores > 0 and computing_cores == 0,
                "ppu_index": ppu_index,
                "ppu_row": pr,
                "ppu_col": pc,
                "active_cores": active_cores,
                "label": f"PPU {ppu_index}",
            }

        global_cycle = cycle_offset + local_cycle
        merged.append(
            CycleSnapshot(
                cycle=global_cycle,
                phase="compute" if any_computing else "done",
                macs=[],
                left_inject=[],
                top_inject=[],
                bottom_output=[],
                active_links=[],
                progress={
                    "logic_cycle": global_cycle,
                    "wave_index": wave_index,
                    "wave_local_cycle": local_cycle,
                    "wave_count": wave_count,
                    "n0": n0,
                    "wave_n": wave_n,
                    "active_core_count": plan.active_core_count,
                    "active_ppu_count": plan.active_ppu_count,
                },
                memory={},
                die_ppuss=die_ppuss,
            )
        )

    return merged


class LogicDieEngine:
    """Simulate OS GEMM on Logic Die hierarchy (32 PPU × 16 core × 16×16 MAC)."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config

    def run(self, m: int, k: int, n: int) -> SimResult:
        chunks = os_n_wave_chunks(n, mac_cols=self.config.cols)
        wave_count = len(chunks)

        all_snapshots: list[CycleSnapshot] = []
        all_tiles: list[CoreTile] = []
        wave_metas: list[dict] = []
        cycle_offset = 0

        for wave_index, (n0, wave_n) in enumerate(chunks):
            plan = compute_os_logic_die_plan(
                m,
                k,
                wave_n,
                n0=n0,
                wave_index=wave_index,
                mac_cols=self.config.cols,
            )
            tile_cycles = {
                tile.key: compute_cycle_count(tile.local_m, tile.local_k, tile.local_n)
                for tile in plan.tiles
            }
            wave_snaps = _merge_os_snapshots(
                plan,
                tile_cycles,
                wave_index=wave_index,
                wave_count=wave_count,
                n0=n0,
                wave_n=wave_n,
                cycle_offset=cycle_offset,
            )
            cycle_start = cycle_offset
            cycle_offset += len(wave_snaps)
            wave_metas.append(
                {
                    "wave_index": wave_index,
                    "n0": n0,
                    "wave_n": wave_n,
                    "cycle_start": cycle_start,
                    "cycle_end": cycle_offset,
                    "active_count": len(plan.tiles),
                    "active_ppu_count": plan.active_ppu_count,
                }
            )
            all_snapshots.extend(wave_snaps)
            all_tiles.extend(plan.tiles)

        combined = LogicDiePlan(
            mac_cols=self.config.cols,
            tiles=all_tiles,
            waves=wave_metas,
            wave_count=wave_count,
        )

        return SimResult(
            config={
                **self.config.to_dict(),
                "hierarchy": "logic_die",
                "die_ppus": DIE_PPU_ROWS * DIE_PPU_COLS,
                "ppu_cores": PPU_CORE_GRID * PPU_CORE_GRID,
                "active_cores": combined.active_core_count,
                "active_ppus": combined.active_ppu_count,
                "wave_count": wave_count,
            },
            dims={"m": m, "k": k, "n": n},
            total_cycles=len(all_snapshots),
            snapshots=all_snapshots,
            tile_plan=combined.to_dict(),
        )

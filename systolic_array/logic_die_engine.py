"""Logic Die engine: OS simulation across 48 PPUs × 16 cores × 16×16 PEs."""

from __future__ import annotations

from systolic_array.config import ArrayConfig
from systolic_array.cycles import compute_cycle_count
from systolic_array.hierarchy import (
    DIE_PPU_COLS,
    DIE_PPU_ROWS,
    PPU_CORE_GRID,
    CoreTile,
    LogicDiePlan,
    compute_os_logic_die_plan,
)
from systolic_array.types import CycleSnapshot, SimResult


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
) -> list[CycleSnapshot]:
    if not plan.tiles:
        return []

    max_cycles = max(tile_cycles.values())
    merged: list[CycleSnapshot] = []

    ppu_active_cores: dict[int, int] = {}
    for tile in plan.tiles:
        ppu_active_cores[tile.ppu_index] = ppu_active_cores.get(tile.ppu_index, 0) + 1

    for cycle in range(max_cycles):
        die_ppuss = _empty_die_grid()
        ppu_computing: dict[int, int] = {}
        any_computing = False

        for tile in plan.tiles:
            if cycle < tile_cycles[tile.key]:
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

        merged.append(
            CycleSnapshot(
                cycle=cycle,
                phase="compute" if any_computing else "done",
                pes=[],
                left_inject=[],
                top_inject=[],
                bottom_output=[],
                active_links=[],
                progress={
                    "logic_cycle": cycle,
                    "active_core_count": plan.active_core_count,
                    "active_ppu_count": plan.active_ppu_count,
                },
                memory={},
                die_ppuss=die_ppuss,
            )
        )

    return merged


class LogicDieEngine:
    """Simulate OS GEMM on Logic Die hierarchy (48 PPU × 16 core × 16×16 PE)."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config

    def run(self, m: int, k: int, n: int) -> SimResult:
        plan = compute_os_logic_die_plan(m, k, n, pe_cols=self.config.cols)

        tile_cycles = {
            tile.key: compute_cycle_count(tile.local_m, tile.local_k, tile.local_n)
            for tile in plan.tiles
        }
        snapshots = _merge_os_snapshots(plan, tile_cycles)

        return SimResult(
            config={
                **self.config.to_dict(),
                "hierarchy": "logic_die",
                "die_ppus": DIE_PPU_ROWS * DIE_PPU_COLS,
                "ppu_cores": PPU_CORE_GRID * PPU_CORE_GRID,
                "active_cores": plan.active_core_count,
                "active_ppus": plan.active_ppu_count,
            },
            dims={"m": m, "k": k, "n": n},
            total_cycles=len(snapshots),
            snapshots=snapshots,
            tile_plan=plan.to_dict(),
        )

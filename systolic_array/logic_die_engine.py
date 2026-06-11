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
from systolic_array.types import CycleSnapshot, MacroArrayState, SimResult


def _core_label(tile: CoreTile) -> str:
    return f"M[0:{tile.local_m}]×N[{tile.n0}:{tile.n0 + tile.local_n}]"


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


def _empty_ppu_core_grid() -> list[list[dict]]:
    return [
        [
            MacroArrayState(active=False, grid_row=r, grid_col=c).to_dict()
            for c in range(PPU_CORE_GRID)
        ]
        for r in range(PPU_CORE_GRID)
    ]


def _merge_os_snapshots(
    plan: LogicDiePlan,
    tile_cycles: dict[str, int],
) -> list[CycleSnapshot]:
    if not plan.tiles:
        return []

    max_cycles = max(tile_cycles.values())
    merged: list[CycleSnapshot] = []

    for cycle in range(max_cycles):
        die_ppuss = _empty_die_grid()
        ppu_core_grids: dict[str, list[list[dict]]] = {
            str(i): _empty_ppu_core_grid() for i in range(plan.active_ppu_count or 1)
        }
        for tile in plan.tiles:
            ppu_core_grids.setdefault(str(tile.ppu_index), _empty_ppu_core_grid())

        any_computing = False

        for tile in plan.tiles:
            ppu_key = str(tile.ppu_index)
            cr, cc = tile.core_row, tile.core_col
            label = _core_label(tile)
            cycles = tile_cycles[tile.key]

            if cycle < cycles:
                any_computing = True
                ppu_core_grids[ppu_key][cr][cc] = MacroArrayState(
                    active=True,
                    computing=True,
                    done=False,
                    grid_row=cr,
                    grid_col=cc,
                    local_m=tile.local_m,
                    local_k=tile.local_k,
                    local_n=tile.local_n,
                    m0=tile.m0,
                    k0=tile.k0,
                    n0=tile.n0,
                    label=label,
                ).to_dict()
            else:
                ppu_core_grids[ppu_key][cr][cc] = MacroArrayState(
                    active=True,
                    computing=False,
                    done=True,
                    grid_row=cr,
                    grid_col=cc,
                    local_m=tile.local_m,
                    local_k=tile.local_k,
                    local_n=tile.local_n,
                    m0=tile.m0,
                    k0=tile.k0,
                    n0=tile.n0,
                    label=label,
                ).to_dict()

        for ppu_index in {t.ppu_index for t in plan.tiles}:
            pr = ppu_index // DIE_PPU_COLS
            pc = ppu_index % DIE_PPU_COLS
            grid = ppu_core_grids[str(ppu_index)]
            ppu_computing = any(
                grid[r][c].get("computing")
                for r in range(PPU_CORE_GRID)
                for c in range(PPU_CORE_GRID)
                if grid[r][c].get("active")
            )
            active_cores = sum(
                1
                for r in range(PPU_CORE_GRID)
                for c in range(PPU_CORE_GRID)
                if grid[r][c].get("active")
            )
            die_ppuss[pr][pc] = {
                "active": True,
                "computing": ppu_computing,
                "done": active_cores > 0 and not ppu_computing,
                "ppu_index": ppu_index,
                "ppu_row": pr,
                "ppu_col": pc,
                "active_cores": active_cores,
                "label": f"PPU {ppu_index}",
            }

        first_ppu = plan.tiles[0].ppu_index
        macro = ppu_core_grids.get(str(first_ppu), _empty_ppu_core_grid())

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
                macro_arrays=macro,
                die_ppuss=die_ppuss,
                ppu_core_grids=ppu_core_grids,
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

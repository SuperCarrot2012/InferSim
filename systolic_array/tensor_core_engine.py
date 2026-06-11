"""Tensor Core engine: orchestrate multiple 16×16 systolic arrays in a 4×4 grid."""

from __future__ import annotations

from systolic_array.config import ArrayConfig
from systolic_array.cycles import compute_cycle_count
from systolic_array.tiling import TC_GRID, ArrayTile, TensorCorePlan, compute_tile_plan
from systolic_array.types import CycleSnapshot, DataflowType, MacroArrayState, SimResult


def _tile_label(tile: ArrayTile, dataflow: DataflowType) -> str:
    if dataflow == DataflowType.OUTPUT_STATIONARY:
        return f"M[{tile.m0}:{tile.m0 + tile.local_m}]×N[{tile.n0}:{tile.n0 + tile.local_n}]"
    return f"K[{tile.k0}:{tile.k0 + tile.local_k}]×N[{tile.n0}:{tile.n0 + tile.local_n}]"


def _empty_macro_grid(plan: TensorCorePlan) -> list[list[dict]]:
    grid: list[list[dict]] = []
    for r in range(plan.grid_rows):
        row: list[dict] = []
        for c in range(plan.grid_cols):
            row.append(MacroArrayState(active=False, grid_row=r, grid_col=c).to_dict())
        grid.append(row)
    return grid


def _merge_snapshots(
    plan: TensorCorePlan,
    dataflow: DataflowType,
    tile_cycles: dict[str, int],
) -> list[CycleSnapshot]:
    if not plan.tiles:
        return []

    max_cycles = max(tile_cycles.values())
    merged: list[CycleSnapshot] = []

    for cycle in range(max_cycles):
        macro: list[list[dict]] = _empty_macro_grid(plan)
        any_computing = False

        for tile in plan.tiles:
            gr, gc = tile.grid_row, tile.grid_col
            label = _tile_label(tile, dataflow)
            cycles = tile_cycles[tile.key]

            if cycle < cycles:
                any_computing = True
                macro[gr][gc] = MacroArrayState(
                    active=True,
                    computing=True,
                    done=False,
                    grid_row=gr,
                    grid_col=gc,
                    local_m=tile.local_m,
                    local_k=tile.local_k,
                    local_n=tile.local_n,
                    m0=tile.m0,
                    k0=tile.k0,
                    n0=tile.n0,
                    label=label,
                ).to_dict()
            else:
                macro[gr][gc] = MacroArrayState(
                    active=True,
                    computing=False,
                    done=True,
                    grid_row=gr,
                    grid_col=gc,
                    local_m=tile.local_m,
                    local_k=tile.local_k,
                    local_n=tile.local_n,
                    m0=tile.m0,
                    k0=tile.k0,
                    n0=tile.n0,
                    label=label,
                ).to_dict()

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
                    "tensor_cycle": cycle,
                    "active_array_count": plan.active_count,
                },
                memory={},
                macro_arrays=macro,
            )
        )

    return merged


class TensorCoreEngine:
    """Simulate a 4×4 Tensor Core by tiling GEMM across systolic arrays."""

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config

    def run(self, m: int, k: int, n: int) -> SimResult:
        plan = compute_tile_plan(
            m,
            k,
            n,
            self.config.dataflow,
            pe_rows=self.config.rows,
            pe_cols=self.config.cols,
            grid=TC_GRID,
        )

        tile_cycles = {
            tile.key: compute_cycle_count(tile.local_m, tile.local_k, tile.local_n)
            for tile in plan.tiles
        }
        snapshots = _merge_snapshots(plan, self.config.dataflow, tile_cycles)

        return SimResult(
            config={
                **self.config.to_dict(),
                "tensor_core_grid": TC_GRID,
                "active_arrays": plan.active_count,
            },
            dims={"m": m, "k": k, "n": n},
            total_cycles=len(snapshots),
            snapshots=snapshots,
            tile_plan=plan.to_dict(),
        )

"""Tensor Core tiling: map GEMM dimensions onto a 4×4 systolic-array grid."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from systolic_array.cycles import compute_cycle_count
from systolic_array.types import DataflowType

TC_GRID = 4
PE_TILE = 16


@dataclass
class ArrayTile:
    """One systolic array's sub-GEMM assignment within the Tensor Core."""

    grid_row: int
    grid_col: int
    m0: int
    k0: int
    n0: int
    local_m: int
    local_k: int
    local_n: int

    @property
    def key(self) -> str:
        return f"{self.grid_row},{self.grid_col}"

    def to_dict(self) -> dict:
        return {
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
            "m0": self.m0,
            "k0": self.k0,
            "n0": self.n0,
            "local_m": self.local_m,
            "local_k": self.local_k,
            "local_n": self.local_n,
            "total_cycles": compute_cycle_count(self.local_m, self.local_k, self.local_n),
        }


@dataclass
class TensorCorePlan:
    grid_rows: int = TC_GRID
    grid_cols: int = TC_GRID
    pe_rows: int = PE_TILE
    pe_cols: int = PE_TILE
    dataflow: str = DataflowType.OUTPUT_STATIONARY.value
    tiles: list[ArrayTile] = field(default_factory=list)

    @property
    def active_count(self) -> int:
        return len(self.tiles)

    def to_dict(self) -> dict:
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "pe_rows": self.pe_rows,
            "pe_cols": self.pe_cols,
            "dataflow": self.dataflow,
            "active_count": self.active_count,
            "tiles": [t.to_dict() for t in self.tiles],
        }


def compute_tile_plan(
    m: int,
    k: int,
    n: int,
    dataflow: DataflowType,
    *,
    pe_rows: int = PE_TILE,
    pe_cols: int = PE_TILE,
    grid: int = TC_GRID,
) -> TensorCorePlan:
    """Split GEMM across up to grid×grid systolic arrays (each pe_rows×pe_cols PEs).

    OS: spatial tiling on M × N (K streams through each array).
    WS: spatial tiling on K × N (M streams through each array).
    """
    tiles: list[ArrayTile] = []

    if dataflow == DataflowType.OUTPUT_STATIONARY:
        m_tiles = math.ceil(m / pe_rows)
        n_tiles = math.ceil(n / pe_cols)
        if m_tiles > grid or n_tiles > grid:
            raise ValueError(
                f"OS tile grid {m_tiles}×{n_tiles} exceeds {grid}×{grid} Tensor Core "
                f"(M={m}, N={n}, tile {pe_rows}×{pe_cols})"
            )
        for mi in range(m_tiles):
            for ni in range(n_tiles):
                m0 = mi * pe_rows
                n0 = ni * pe_cols
                tiles.append(
                    ArrayTile(
                        grid_row=mi,
                        grid_col=ni,
                        m0=m0,
                        k0=0,
                        n0=n0,
                        local_m=min(pe_rows, m - m0),
                        local_k=k,
                        local_n=min(pe_cols, n - n0),
                    )
                )
    elif dataflow == DataflowType.WEIGHT_STATIONARY:
        k_tiles = math.ceil(k / pe_rows)
        n_tiles = math.ceil(n / pe_cols)
        if k_tiles > grid or n_tiles > grid:
            raise ValueError(
                f"WS tile grid {k_tiles}×{n_tiles} exceeds {grid}×{grid} Tensor Core "
                f"(K={k}, N={n}, tile {pe_rows}×{pe_cols})"
            )
        for ki in range(k_tiles):
            for ni in range(n_tiles):
                k0 = ki * pe_rows
                n0 = ni * pe_cols
                tiles.append(
                    ArrayTile(
                        grid_row=ki,
                        grid_col=ni,
                        m0=0,
                        k0=k0,
                        n0=n0,
                        local_m=m,
                        local_k=min(pe_rows, k - k0),
                        local_n=min(pe_cols, n - n0),
                    )
                )
    else:
        raise NotImplementedError(f"Tiling not implemented for {dataflow}")

    return TensorCorePlan(
        pe_rows=pe_rows,
        pe_cols=pe_cols,
        dataflow=dataflow.value,
        tiles=tiles,
    )

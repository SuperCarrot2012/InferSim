"""Tensor Core tiling: map GEMM dimensions onto a 4×4 systolic-array grid."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from systolic_array_model.cycles import compute_cycle_count
from systolic_array_model.types import DataflowType

TC_GRID = 4
MAC_TILE = 16


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
    mac_rows: int = MAC_TILE
    mac_cols: int = MAC_TILE
    dataflow: str = DataflowType.OUTPUT_STATIONARY.value
    tiles: list[ArrayTile] = field(default_factory=list)

    @property
    def active_count(self) -> int:
        return len(self.tiles)

    def to_dict(self) -> dict:
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "mac_rows": self.mac_rows,
            "mac_cols": self.mac_cols,
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
    mac_rows: int = MAC_TILE,
    mac_cols: int = MAC_TILE,
    grid: int = TC_GRID,
) -> TensorCorePlan:
    """Split GEMM across up to grid×grid systolic arrays (each mac_rows×mac_cols MAC units).

    OS: spatial tiling on M × N (K streams through each array).
    WS: spatial tiling on K × N (M streams through each array).
    """
    tiles: list[ArrayTile] = []

    if dataflow == DataflowType.OUTPUT_STATIONARY:
        m_tiles = math.ceil(m / mac_rows)
        n_tiles = math.ceil(n / mac_cols)
        if m_tiles > grid or n_tiles > grid:
            raise ValueError(
                f"OS tile grid {m_tiles}×{n_tiles} exceeds {grid}×{grid} Tensor Core "
                f"(M={m}, N={n}, tile {mac_rows}×{mac_cols})"
            )
        for mi in range(m_tiles):
            for ni in range(n_tiles):
                m0 = mi * mac_rows
                n0 = ni * mac_cols
                tiles.append(
                    ArrayTile(
                        grid_row=mi,
                        grid_col=ni,
                        m0=m0,
                        k0=0,
                        n0=n0,
                        local_m=min(mac_rows, m - m0),
                        local_k=k,
                        local_n=min(mac_cols, n - n0),
                    )
                )
    elif dataflow == DataflowType.WEIGHT_STATIONARY:
        k_tiles = math.ceil(k / mac_rows)
        n_tiles = math.ceil(n / mac_cols)
        if k_tiles > grid or n_tiles > grid:
            raise ValueError(
                f"WS tile grid {k_tiles}×{n_tiles} exceeds {grid}×{grid} Tensor Core "
                f"(K={k}, N={n}, tile {mac_rows}×{mac_cols})"
            )
        for ki in range(k_tiles):
            for ni in range(n_tiles):
                k0 = ki * mac_rows
                n0 = ni * mac_cols
                tiles.append(
                    ArrayTile(
                        grid_row=ki,
                        grid_col=ni,
                        m0=0,
                        k0=k0,
                        n0=n0,
                        local_m=m,
                        local_k=min(mac_rows, k - k0),
                        local_n=min(mac_cols, n - n0),
                    )
                )
    else:
        raise NotImplementedError(f"Tiling not implemented for {dataflow}")

    return TensorCorePlan(
        mac_rows=mac_rows,
        mac_cols=mac_cols,
        dataflow=dataflow.value,
        tiles=tiles,
    )

"""TPU compute hierarchy: Logic Die → PPU → Systolic Core → PE."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from systolic_array_model.cycles import compute_cycle_count
from systolic_array_model.types import DataflowType

# Layer 1 — Logic Die (32 PPU)
DIE_PPU_ROWS = 4
DIE_PPU_COLS = 8
DIE_PPU_COUNT = DIE_PPU_ROWS * DIE_PPU_COLS  # 32

# Layer 2 — PPU (Tensor Core)
PPU_CORE_GRID = 4
PPU_CORE_COUNT = PPU_CORE_GRID * PPU_CORE_GRID  # 16

# Layer 3 — Systolic Core
CORE_PE = 16

OS_M_MAX = CORE_PE
OS_N_MAX = DIE_PPU_COUNT * PPU_CORE_COUNT * CORE_PE  # 32 × 16 × 16 = 8192
# OS: K streams through each core (no spatial K tiling); API upper bound only.
OS_K_MAX = 65536


@dataclass
class CoreTile:
    """One systolic core's OS sub-GEMM (M×N slice; K streams)."""

    ppu_index: int
    ppu_row: int
    ppu_col: int
    core_row: int
    core_col: int
    m0: int
    k0: int
    n0: int
    local_m: int
    local_k: int
    local_n: int

    @property
    def key(self) -> str:
        return f"{self.ppu_index},{self.core_row},{self.core_col}"

    def to_dict(self) -> dict:
        return {
            "ppu_index": self.ppu_index,
            "ppu_row": self.ppu_row,
            "ppu_col": self.ppu_col,
            "core_row": self.core_row,
            "core_col": self.core_col,
            "grid_row": self.core_row,
            "grid_col": self.core_col,
            "m0": self.m0,
            "k0": self.k0,
            "n0": self.n0,
            "local_m": self.local_m,
            "local_k": self.local_k,
            "local_n": self.local_n,
            "total_cycles": compute_cycle_count(self.local_m, self.local_k, self.local_n),
        }


@dataclass
class LogicDiePlan:
    """OS tiling plan across Logic Die → PPU → Core."""

    die_rows: int = DIE_PPU_ROWS
    die_cols: int = DIE_PPU_COLS
    ppu_core_grid: int = PPU_CORE_GRID
    pe_rows: int = CORE_PE
    pe_cols: int = CORE_PE
    dataflow: str = DataflowType.OUTPUT_STATIONARY.value
    tiles: list[CoreTile] = field(default_factory=list)

    @property
    def active_core_count(self) -> int:
        return len(self.tiles)

    @property
    def active_ppu_count(self) -> int:
        return len({t.ppu_index for t in self.tiles})

    def to_dict(self) -> dict:
        return {
            "hierarchy": "logic_die",
            "die_rows": self.die_rows,
            "die_cols": self.die_cols,
            "ppu_count": DIE_PPU_COUNT,
            "ppu_core_grid": self.ppu_core_grid,
            "pe_rows": self.pe_rows,
            "pe_cols": self.pe_cols,
            "dataflow": self.dataflow,
            "active_count": self.active_core_count,
            "active_ppu_count": self.active_ppu_count,
            "tiles": [t.to_dict() for t in self.tiles],
        }


def _ppu_coords(ppu_index: int) -> tuple[int, int]:
    return ppu_index // DIE_PPU_COLS, ppu_index % DIE_PPU_COLS


def compute_os_logic_die_plan(
    m: int,
    k: int,
    n: int,
    *,
    pe_cols: int = CORE_PE,
) -> LogicDiePlan:
    """OS: M must fit in one core (1–16); split N across cores then PPUs.

    Priority: partition N into 16-wide slices mapped to systolic cores inside
    each PPU (4×4), then spill to additional PPUs on the Logic Die (4×8).
    """
    if m < 1 or m > OS_M_MAX:
        raise ValueError(
            f"OS mode: M must be in [1, {OS_M_MAX}], got M={m}"
        )

    n_slices = math.ceil(n / pe_cols)
    ppus_needed = math.ceil(n_slices / PPU_CORE_COUNT)

    if ppus_needed > DIE_PPU_COUNT:
        raise ValueError(
            f"OS mode: N={n} requires {ppus_needed} PPUs "
            f"(max {DIE_PPU_COUNT} on Logic Die)"
        )

    tiles: list[CoreTile] = []
    for slice_idx in range(n_slices):
        ppu_index = slice_idx // PPU_CORE_COUNT
        core_idx = slice_idx % PPU_CORE_COUNT
        core_row, core_col = divmod(core_idx, PPU_CORE_GRID)
        ppu_row, ppu_col = _ppu_coords(ppu_index)
        n0 = slice_idx * pe_cols

        tiles.append(
            CoreTile(
                ppu_index=ppu_index,
                ppu_row=ppu_row,
                ppu_col=ppu_col,
                core_row=core_row,
                core_col=core_col,
                m0=0,
                k0=0,
                n0=n0,
                local_m=m,
                local_k=k,
                local_n=min(pe_cols, n - n0),
            )
        )

    return LogicDiePlan(pe_cols=pe_cols, tiles=tiles)

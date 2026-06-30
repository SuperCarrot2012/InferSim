"""TPU compute hierarchy: Logic Die → PPU → Systolic Core → MAC unit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from systolic_array_model.cycles import compute_cycle_count, compute_os_cycle_count
from systolic_array_model.types import DataflowType

# Layer 1 — Logic Die (32 PPU)
DIE_PPU_ROWS = 4
DIE_PPU_COLS = 8
DIE_PPU_COUNT = DIE_PPU_ROWS * DIE_PPU_COLS  # 32

# Layer 2 — PPU (Tensor Core)
PPU_CORE_GRID = 4
PPU_CORE_COUNT = PPU_CORE_GRID * PPU_CORE_GRID  # 16

# Layer 3 — Systolic Core
CORE_MAC = 16

OS_M_MAX = CORE_MAC
OS_N_MAX = DIE_PPU_COUNT * PPU_CORE_COUNT * CORE_MAC  # 32 × 16 × 16 = 8192
# Max N columns mappable on Logic Die in one wave.
OS_N_WAVE_MAX = OS_N_MAX
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
    wave_index: int = 0
    head_index: int = 0

    @property
    def key(self) -> str:
        return f"{self.wave_index},{self.ppu_index},{self.core_row},{self.core_col}"

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
            "wave_index": self.wave_index,
            "head_index": self.head_index,
            "total_cycles": compute_os_cycle_count(self.local_m, self.local_k, self.local_n),
        }


@dataclass
class LogicDiePlan:
    """OS tiling plan across Logic Die → PPU → Core."""

    die_rows: int = DIE_PPU_ROWS
    die_cols: int = DIE_PPU_COLS
    ppu_core_grid: int = PPU_CORE_GRID
    mac_rows: int = CORE_MAC
    mac_cols: int = CORE_MAC
    dataflow: str = DataflowType.OUTPUT_STATIONARY.value
    tiles: list[CoreTile] = field(default_factory=list)
    waves: list[dict] = field(default_factory=list)
    wave_count: int = 1

    @property
    def active_core_count(self) -> int:
        if self.waves:
            return max(w["active_count"] for w in self.waves)
        return len(self.tiles)

    @property
    def active_ppu_count(self) -> int:
        if self.waves:
            return max(w["active_ppu_count"] for w in self.waves)
        return len({t.ppu_index for t in self.tiles})

    def to_dict(self) -> dict:
        return {
            "hierarchy": "logic_die",
            "die_rows": self.die_rows,
            "die_cols": self.die_cols,
            "ppu_count": DIE_PPU_COUNT,
            "ppu_core_grid": self.ppu_core_grid,
            "mac_rows": self.mac_rows,
            "mac_cols": self.mac_cols,
            "dataflow": self.dataflow,
            "active_count": self.active_core_count,
            "active_ppu_count": self.active_ppu_count,
            "wave_count": self.wave_count,
            "os_n_wave_max": OS_N_WAVE_MAX,
            "waves": self.waves,
            "tiles": [t.to_dict() for t in self.tiles],
        }


def _ppu_coords(ppu_index: int) -> tuple[int, int]:
    return ppu_index // DIE_PPU_COLS, ppu_index % DIE_PPU_COLS


def os_n_wave_chunks(n: int, *, mac_cols: int = CORE_MAC) -> list[tuple[int, int]]:
    """Split global N into sequential waves that each fit on one Logic Die."""
    max_per_wave = DIE_PPU_COUNT * PPU_CORE_COUNT * mac_cols
    chunks: list[tuple[int, int]] = []
    n0 = 0
    while n0 < n:
        wave_n = min(max_per_wave, n - n0)
        chunks.append((n0, wave_n))
        n0 += wave_n
    return chunks


def os_n_wave_count(n: int, *, mac_cols: int = CORE_MAC) -> int:
    return len(os_n_wave_chunks(n, mac_cols=mac_cols))


def os_ppu_n_wave_chunks(n: int, *, mac_cols: int = CORE_MAC) -> list[tuple[int, int]]:
    """Split global N into waves that fit on one PPU (16 cores × mac_cols)."""
    max_per_ppu = PPU_CORE_COUNT * mac_cols
    chunks: list[tuple[int, int]] = []
    n0 = 0
    while n0 < n:
        wave_n = min(max_per_ppu, n - n0)
        chunks.append((n0, wave_n))
        n0 += wave_n
    return chunks


def os_ppu_n_wave_count(n: int, *, mac_cols: int = CORE_MAC) -> int:
    return len(os_ppu_n_wave_chunks(n, mac_cols=mac_cols))


def ppus_needed_for_n(n: int, *, mac_cols: int = CORE_MAC) -> int:
    """PPUs required to map N columns on Logic Die (one head / one GEMM)."""
    n_slices = math.ceil(n / mac_cols)
    return math.ceil(n_slices / PPU_CORE_COUNT)


def one_head_per_ppu_layout(batch: int) -> tuple[int, int]:
    """Return (heads_per_wave, batch_waves) — one Q head per PPU, all heads parallel."""
    if batch > DIE_PPU_COUNT:
        raise ValueError(
            f"one head per PPU: batch={batch} exceeds {DIE_PPU_COUNT} PPUs"
        )
    return batch, 1


def compute_os_single_ppu_plan(
    m: int,
    k: int,
    n: int,
    *,
    ppu_index: int,
    n0: int = 0,
    wave_index: int = 0,
    head_index: int = 0,
    mac_cols: int = CORE_MAC,
) -> LogicDiePlan:
    """OS tile plan for one head on a single PPU (N split across its 16 cores only)."""
    if m < 1 or m > OS_M_MAX:
        raise ValueError(f"OS mode: M must be in [1, {OS_M_MAX}], got M={m}")
    if ppu_index < 0 or ppu_index >= DIE_PPU_COUNT:
        raise ValueError(f"ppu_index must be in [0, {DIE_PPU_COUNT}), got {ppu_index}")

    n_slices = math.ceil(n / mac_cols)
    if n_slices > PPU_CORE_COUNT:
        raise ValueError(
            f"OS single-PPU: N={n} needs {n_slices} cores but one PPU has {PPU_CORE_COUNT}"
        )

    ppu_row, ppu_col = _ppu_coords(ppu_index)
    tiles: list[CoreTile] = []
    for slice_idx in range(n_slices):
        core_idx = slice_idx
        core_row, core_col = divmod(core_idx, PPU_CORE_GRID)
        slice_n0 = slice_idx * mac_cols

        tiles.append(
            CoreTile(
                ppu_index=ppu_index,
                ppu_row=ppu_row,
                ppu_col=ppu_col,
                core_row=core_row,
                core_col=core_col,
                m0=0,
                k0=0,
                n0=n0 + slice_n0,
                local_m=m,
                local_k=k,
                local_n=min(mac_cols, n - slice_n0),
                wave_index=wave_index,
                head_index=head_index,
            )
        )

    return LogicDiePlan(mac_cols=mac_cols, tiles=tiles)



def compute_os_logic_die_plan(
    m: int,
    k: int,
    n: int,
    *,
    n0: int = 0,
    wave_index: int = 0,
    head_index: int = 0,
    ppu_offset: int = 0,
    mac_cols: int = CORE_MAC,
) -> LogicDiePlan:
    """OS: M must fit in one core (1–16); split N across cores then PPUs.

    ``n`` is the column count for this wave; ``n0`` is the global N offset.
    When total N exceeds ``OS_N_WAVE_MAX``, run multiple waves via ``os_n_wave_chunks``.
    """
    if m < 1 or m > OS_M_MAX:
        raise ValueError(
            f"OS mode: M must be in [1, {OS_M_MAX}], got M={m}"
        )

    n_slices = math.ceil(n / mac_cols)
    ppus_needed = math.ceil(n_slices / PPU_CORE_COUNT)
    ppu_limit = DIE_PPU_COUNT - ppu_offset

    if ppus_needed > ppu_limit:
        raise ValueError(
            f"OS mode: N={n} requires {ppus_needed} PPUs from offset {ppu_offset} "
            f"(max {ppu_limit} available)"
        )

    tiles: list[CoreTile] = []
    for slice_idx in range(n_slices):
        ppu_index = ppu_offset + slice_idx // PPU_CORE_COUNT
        core_idx = slice_idx % PPU_CORE_COUNT
        core_row, core_col = divmod(core_idx, PPU_CORE_GRID)
        ppu_row, ppu_col = _ppu_coords(ppu_index)
        slice_n0 = slice_idx * mac_cols

        tiles.append(
            CoreTile(
                ppu_index=ppu_index,
                ppu_row=ppu_row,
                ppu_col=ppu_col,
                core_row=core_row,
                core_col=core_col,
                m0=0,
                k0=0,
                n0=n0 + slice_n0,
                local_m=m,
                local_k=k,
                local_n=min(mac_cols, n - slice_n0),
                wave_index=wave_index,
                head_index=head_index,
            )
        )

    return LogicDiePlan(mac_cols=mac_cols, tiles=tiles)

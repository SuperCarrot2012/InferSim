"""Shared datatypes for systolic array simulation snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataflowType(str, Enum):
    WEIGHT_STATIONARY = "weight_stationary"
    OUTPUT_STATIONARY = "output_stationary"


class PEPhase(str, Enum):
    IDLE = "idle"
    LOAD_WEIGHT = "load_weight"
    LOAD_PSUM = "load_psum"
    COMPUTE = "compute"
    DRAIN = "drain"


class SimPhase(str, Enum):
    WEIGHT_LOAD = "weight_load"
    PSUM_STORE = "psum_store"
    COMPUTE = "compute"
    DRAIN = "drain"
    DONE = "done"


@dataclass
class LinkAnim:
    """Animated data movement between PEs for one cycle."""

    from_row: int
    from_col: int
    to_row: int
    to_col: int
    label: str
    direction: str  # "right" | "down"


@dataclass
class PESnapshot:
    row: int
    col: int
    phase: str
    w_coord: list[int] | None = None  # W[k, n]
    a_coord: list[int] | None = None  # A[m, k]
    p_coord: list[int] | None = None  # P[m, n] partial sum
    has_act: bool = False
    has_weight: bool = False
    has_psum: bool = False
    writeback: bool = False
    in_tile: bool = True


@dataclass
class CycleSnapshot:
    cycle: int
    phase: str
    pes: list[list[PESnapshot]]
    left_inject: list[str | None]
    bottom_output: list[str | None]
    active_links: list[LinkAnim]
    progress: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    top_inject: list[str | None] = field(default_factory=list)
    macro_arrays: list[list[dict[str, Any]]] | None = None
    micro_tiles: dict[str, CycleSnapshot] | None = None
    die_ppuss: list[list[dict[str, Any]]] | None = None
    ppu_core_grids: dict[str, list[list[dict[str, Any]]]] | None = None


@dataclass
class MacroArrayState:
    active: bool
    computing: bool = False
    done: bool = False
    grid_row: int = 0
    grid_col: int = 0
    local_m: int = 0
    local_k: int = 0
    local_n: int = 0
    m0: int = 0
    k0: int = 0
    n0: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "computing": self.computing,
            "done": self.done,
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
            "local_m": self.local_m,
            "local_k": self.local_k,
            "local_n": self.local_n,
            "m0": self.m0,
            "k0": self.k0,
            "n0": self.n0,
            "label": self.label,
        }


@dataclass
class SimResult:
    config: dict[str, Any]
    dims: dict[str, int]
    total_cycles: int
    snapshots: list[CycleSnapshot]
    tile_plan: dict[str, Any] = field(default_factory=dict)
    micro_store: dict[str, list[CycleSnapshot]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "dims": self.dims,
            "total_cycles": self.total_cycles,
            "tile_plan": self.tile_plan,
            "snapshots": [_macro_snapshot_to_dict(s) for s in self.snapshots],
            "micro_store": {
                k: [_micro_tile_to_dict(s) for s in cycles]
                for k, cycles in self.micro_store.items()
            },
        }


def _micro_tile_to_dict(snapshot: CycleSnapshot) -> dict[str, Any]:
    return {
        "phase": snapshot.phase,
        "pes": [
            [
                {
                    "row": pe.row,
                    "col": pe.col,
                    "phase": pe.phase,
                    "w_coord": pe.w_coord,
                    "a_coord": pe.a_coord,
                    "p_coord": pe.p_coord,
                    "has_act": pe.has_act,
                    "has_weight": pe.has_weight,
                    "has_psum": pe.has_psum,
                    "writeback": pe.writeback,
                    "in_tile": pe.in_tile,
                }
                for pe in row
            ]
            for row in snapshot.pes
        ],
        "left_inject": snapshot.left_inject,
        "top_inject": snapshot.top_inject,
        "bottom_output": snapshot.bottom_output,
        "active_links": [
            {
                "from_row": link.from_row,
                "from_col": link.from_col,
                "to_row": link.to_row,
                "to_col": link.to_col,
                "label": link.label,
                "direction": link.direction,
            }
            for link in snapshot.active_links
        ],
        "progress": snapshot.progress,
        "memory": snapshot.memory,
    }


def build_micro_store(
    tile_runs: list[tuple[Any, list[CycleSnapshot]]],
) -> dict[str, list[CycleSnapshot]]:
    """Per-tile full PE snapshots, keyed by tile.key (on-demand fetch)."""
    store: dict[str, list[CycleSnapshot]] = {}
    for tile, snaps in tile_runs:
        store[tile.key] = snaps
    return store


def _macro_snapshot_to_dict(snapshot: CycleSnapshot) -> dict[str, Any]:
    """Lightweight per-cycle snapshot (Logic Die / PPU views only, no PE grid)."""
    d: dict[str, Any] = {
        "cycle": snapshot.cycle,
        "phase": snapshot.phase,
        "progress": snapshot.progress,
        "memory": snapshot.memory,
    }
    if snapshot.macro_arrays is not None:
        d["macro_arrays"] = snapshot.macro_arrays
    if snapshot.die_ppuss is not None:
        d["die_ppuss"] = snapshot.die_ppuss
    if snapshot.ppu_core_grids is not None:
        d["ppu_core_grids"] = snapshot.ppu_core_grids
    return d


def _snapshot_to_dict(snapshot: CycleSnapshot) -> dict[str, Any]:
    d: dict[str, Any] = {
        "cycle": snapshot.cycle,
        "phase": snapshot.phase,
        "pes": [
            [
                {
                    "row": pe.row,
                    "col": pe.col,
                    "phase": pe.phase,
                    "w_coord": pe.w_coord,
                    "a_coord": pe.a_coord,
                    "p_coord": pe.p_coord,
                    "has_act": pe.has_act,
                    "has_weight": pe.has_weight,
                    "has_psum": pe.has_psum,
                    "writeback": pe.writeback,
                    "in_tile": pe.in_tile,
                }
                for pe in row
            ]
            for row in snapshot.pes
        ],
        "left_inject": snapshot.left_inject,
        "top_inject": snapshot.top_inject,
        "bottom_output": snapshot.bottom_output,
        "active_links": [
            {
                "from_row": link.from_row,
                "from_col": link.from_col,
                "to_row": link.to_row,
                "to_col": link.to_col,
                "label": link.label,
                "direction": link.direction,
            }
            for link in snapshot.active_links
        ],
        "progress": snapshot.progress,
        "memory": snapshot.memory,
    }
    if snapshot.macro_arrays is not None:
        d["macro_arrays"] = snapshot.macro_arrays
    if snapshot.die_ppuss is not None:
        d["die_ppuss"] = snapshot.die_ppuss
    if snapshot.ppu_core_grids is not None:
        d["ppu_core_grids"] = snapshot.ppu_core_grids
    if snapshot.micro_tiles is not None:
        d["micro_tiles"] = {
            k: _micro_tile_to_dict(v) for k, v in snapshot.micro_tiles.items()
        }
    return d

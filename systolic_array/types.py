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


@dataclass
class SimResult:
    config: dict[str, Any]
    dims: dict[str, int]
    total_cycles: int
    snapshots: list[CycleSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "dims": self.dims,
            "total_cycles": self.total_cycles,
            "snapshots": [_snapshot_to_dict(s) for s in self.snapshots],
        }


def _snapshot_to_dict(snapshot: CycleSnapshot) -> dict[str, Any]:
    return {
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

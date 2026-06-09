"""Systolic array hardware configuration."""

from __future__ import annotations

from dataclasses import dataclass

from systolic_array.types import DataflowType


@dataclass
class ArrayConfig:
    """Hardware parameters for the systolic array simulator.

    mac_latency: number of pipeline stages in the MAC unit. With latency=1,
    operands enter and the updated accumulator is visible in the same cycle
    (combinational MAC). With latency>1, each MAC takes that many cycles
    through the pipeline before the result is visible downstream.
    """

    rows: int = 16
    cols: int = 16
    dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY
    mac_latency: int = 1
    weight_load_cycles: int = 1

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Array dimensions must be positive")
        if self.mac_latency < 1:
            raise ValueError("mac_latency must be >= 1")

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "dataflow": self.dataflow.value,
            "mac_latency": self.mac_latency,
            "weight_load_cycles": self.weight_load_cycles,
        }

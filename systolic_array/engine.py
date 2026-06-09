"""Cycle-accurate simulation engine."""

from __future__ import annotations

from systolic_array.array import SystolicArray
from systolic_array.config import ArrayConfig
from systolic_array.dataflow.output_stationary import OutputStationaryDataflow
from systolic_array.dataflow.weight_stationary import WeightStationaryDataflow
from systolic_array.types import DataflowType, SimResult


class CycleEngine:
    def __init__(self, config: ArrayConfig) -> None:
        self.config = config
        self.array = SystolicArray(config)

    def run(self, m: int, k: int, n: int) -> SimResult:
        self.array.reset()
        if self.config.dataflow == DataflowType.WEIGHT_STATIONARY:
            dataflow = WeightStationaryDataflow(self.config)
        elif self.config.dataflow == DataflowType.OUTPUT_STATIONARY:
            dataflow = OutputStationaryDataflow(self.config)
        else:
            raise NotImplementedError(
                f"Dataflow {self.config.dataflow} not yet implemented"
            )
        snapshots = dataflow.run(self.array, m, k, n)

        return SimResult(
            config=self.config.to_dict(),
            dims={"m": m, "k": k, "n": n},
            total_cycles=len(snapshots),
            snapshots=snapshots,
        )

"""Cycle-accurate simulation engine."""

from __future__ import annotations

from systolic_array.array import SystolicArray
from systolic_array.config import ArrayConfig
from systolic_array.dataflow.output_stationary import OutputStationaryDataflow
from systolic_array.dataflow.weight_stationary import WeightStationaryDataflow
from systolic_array.cycles import compute_cycle_count
from systolic_array.types import CycleSnapshot, DataflowType, SimResult


class CycleEngine:
    def __init__(self, config: ArrayConfig) -> None:
        self.config = config
        self.array = SystolicArray(config)

    def run(
        self,
        m: int,
        k: int,
        n: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> SimResult:
        self.array.reset()
        if self.config.dataflow == DataflowType.WEIGHT_STATIONARY:
            dataflow = WeightStationaryDataflow(self.config)
        elif self.config.dataflow == DataflowType.OUTPUT_STATIONARY:
            dataflow = OutputStationaryDataflow(self.config)
        else:
            raise NotImplementedError(
                f"Dataflow {self.config.dataflow} not yet implemented"
            )
        snapshots = dataflow.run(
            self.array, m, k, n, m0=m0, k0=k0, n0=n0
        )

        return SimResult(
            config=self.config.to_dict(),
            dims={"m": m, "k": k, "n": n},
            total_cycles=len(snapshots),
            snapshots=snapshots,
        )

    def snapshot_at_cycle(
        self,
        m: int,
        k: int,
        n: int,
        cycle: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> CycleSnapshot:
        self.array.reset()
        if self.config.dataflow == DataflowType.WEIGHT_STATIONARY:
            dataflow = WeightStationaryDataflow(self.config)
        elif self.config.dataflow == DataflowType.OUTPUT_STATIONARY:
            dataflow = OutputStationaryDataflow(self.config)
        else:
            raise NotImplementedError(
                f"Dataflow {self.config.dataflow} not yet implemented"
            )
        return dataflow.snapshot_at_cycle(
            self.array, m, k, n, cycle, m0=m0, k0=k0, n0=n0
        )

    def snapshot_after_drain(
        self,
        m: int,
        k: int,
        n: int,
        *,
        m0: int = 0,
        k0: int = 0,
        n0: int = 0,
    ) -> CycleSnapshot:
        self.array.reset()
        if self.config.dataflow == DataflowType.OUTPUT_STATIONARY:
            dataflow = OutputStationaryDataflow(self.config)
            return dataflow.snapshot_after_drain(
                self.array, m, k, n, m0=m0, k0=k0, n0=n0
            )
        # WS: no separate drain frame; hold last compute snapshot.
        total = compute_cycle_count(m, k, n)
        return self.snapshot_at_cycle(
            m, k, n, max(0, total - 1), m0=m0, k0=k0, n0=n0
        )

    @staticmethod
    def cycle_count(m: int, k: int, n: int) -> int:
        return compute_cycle_count(m, k, n)

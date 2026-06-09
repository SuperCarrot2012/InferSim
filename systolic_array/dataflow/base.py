"""Weight-stationary dataflow for C = A @ B."""

from __future__ import annotations

from abc import ABC, abstractmethod

from systolic_array.array import SystolicArray
from systolic_array.config import ArrayConfig
from systolic_array.types import CycleSnapshot, SimPhase


class DataflowStrategy(ABC):
    @abstractmethod
    def run(
        self,
        array: SystolicArray,
        a: list[list[float]],
        b: list[list[float]],
    ) -> list[CycleSnapshot]:
        raise NotImplementedError


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    n = len(b[0])
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for t in range(k):
                c[i][j] += a[i][t] * b[t][j]
    return c

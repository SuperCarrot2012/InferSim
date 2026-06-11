"""Cycle-count helpers for tiled GEMM simulation."""

from __future__ import annotations


def compute_cycle_count(local_m: int, local_k: int, local_n: int) -> int:
    """Compute-phase length for one tile (OS and WS use the same wavefront formula)."""
    return max(0, local_m + local_k + local_n - 3) + 1

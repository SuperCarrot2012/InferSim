"""Cycle-count helpers for tiled GEMM simulation."""

from __future__ import annotations


def compute_cycle_count(local_m: int, local_k: int, local_n: int) -> int:
    """Wavefront tile length through the last MAC (WS tiles; OS compute phase)."""
    return max(0, local_m + local_k + local_n - 3) + 1


def compute_os_cycle_count(local_m: int, local_k: int, local_n: int) -> int:
    """OS tile length: last MAC cycle plus one registered writeback cycle."""
    return compute_cycle_count(local_m, local_k, local_n) + 1


def os_max_local_cycle(local_m: int, local_k: int, local_n: int) -> int:
    """Last local cycle index for an OS tile (includes writeback)."""
    return max(0, local_m + local_k + local_n - 2)

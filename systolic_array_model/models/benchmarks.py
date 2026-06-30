"""Reference cycle counts for external hardware performance comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BENCHMARK_FILE = Path(__file__).with_name("benchmark_cycles.json")


def _load_benchmark_file() -> dict[str, Any]:
    with _BENCHMARK_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_benchmarks(*, model_id: str | None = None) -> list[dict[str, Any]]:
    """Return benchmark entries, optionally filtered by model_id."""
    entries = _load_benchmark_file().get("benchmarks", [])
    if model_id is None:
        return entries
    return [entry for entry in entries if entry.get("model_id") == model_id]


def get_benchmark_cycles(benchmark_id: str, gemm_id: str) -> int | None:
    """Look up reference cycles for one GEMM op under a benchmark device."""
    for entry in _load_benchmark_file().get("benchmarks", []):
        if entry.get("id") != benchmark_id:
            continue
        cycles = entry.get("gemm_cycles", {})
        value = cycles.get(gemm_id)
        return int(value) if value is not None else None
    return None

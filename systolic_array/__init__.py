"""Systolic array cycle-accurate dataflow simulator."""

from systolic_array.config import ArrayConfig
from systolic_array.simulator import run_simulation, result_to_dict

__all__ = ["ArrayConfig", "run_simulation", "result_to_dict"]

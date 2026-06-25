"""Systolic array cycle-accurate dataflow simulator (model-driven variant)."""

from systolic_array_model.config import ArrayConfig
from systolic_array_model.simulator import run_simulation, result_to_dict

__all__ = ["ArrayConfig", "run_simulation", "result_to_dict"]

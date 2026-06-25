"""Model GEMM operation registry (Llama3-8B decode, tp=1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from systolic_array_model.hierarchy import (
    CORE_PE,
    DIE_PPU_COUNT,
    OS_K_MAX,
    OS_M_MAX,
    PPU_CORE_COUNT,
)


@dataclass(frozen=True)
class GemmOp:
    id: str
    label: str
    m: int
    k: int
    n: int
    note: str = ""

    def ppus_needed(self) -> int:
        n_slices = math.ceil(self.n / CORE_PE)
        return math.ceil(n_slices / PPU_CORE_COUNT)

    def fits_die(self) -> bool:
        if self.m < 1 or self.m > OS_M_MAX:
            return False
        if self.k < 1 or self.k > OS_K_MAX:
            return False
        return self.ppus_needed() <= DIE_PPU_COUNT

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "note": self.note,
            "ppus_needed": self.ppus_needed(),
            "fits_die": self.fits_die(),
        }


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    gemm_ops: list[GemmOp] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "gemm_ops": [op.to_dict() for op in self.gemm_ops],
        }


# Llama3-8B (hidden=4096, heads=32, kv_heads=8, intermediate=14336, vocab=128256)
# Decode step: batch=1, seq_len=1 → M=1
_HIDDEN = 4096
_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = _HIDDEN // _HEADS  # 128
_INTERMEDIATE = 14336
_VOCAB = 128256
_DECODE_M = 1

LLAMA3_8B = ModelSpec(
    id="llama3-8b",
    label="Llama3-8B",
    description="Decode 单 token 步 (batch=1, seq=1, tp=1)",
    gemm_ops=[
        GemmOp("q_proj", "Q_Proj", _DECODE_M, _HIDDEN, _HEADS * _HEAD_DIM),
        GemmOp("k_proj", "K_Proj", _DECODE_M, _HIDDEN, _KV_HEADS * _HEAD_DIM),
        GemmOp("v_proj", "V_Proj", _DECODE_M, _HIDDEN, _KV_HEADS * _HEAD_DIM),
        GemmOp("o_proj", "O_Proj", _DECODE_M, _HIDDEN, _HIDDEN),
        GemmOp("gate_proj", "Gate_Proj", _DECODE_M, _HIDDEN, _INTERMEDIATE),
        GemmOp("up_proj", "Up_Proj", _DECODE_M, _HIDDEN, _INTERMEDIATE),
        GemmOp("down_proj", "Down_Proj", _DECODE_M, _INTERMEDIATE, _HIDDEN),
        GemmOp("lm_head", "LM_Head", _DECODE_M, _HIDDEN, _VOCAB),
    ],
)

_MODELS: dict[str, ModelSpec] = {
    LLAMA3_8B.id: LLAMA3_8B,
}


def get_models_catalog() -> dict:
    return {
        "models": [spec.to_dict() for spec in _MODELS.values()],
        "default_model_id": LLAMA3_8B.id,
        "default_gemm_id": "q_proj",
        "hardware": {
            "die_ppu_count": DIE_PPU_COUNT,
            "os_m_max": OS_M_MAX,
            "os_k_max": OS_K_MAX,
            "os_n_max": DIE_PPU_COUNT * PPU_CORE_COUNT * CORE_PE,
            "memory": "sram_only",
        },
    }


def get_gemm_op(model_id: str, gemm_id: str) -> GemmOp:
    model = _MODELS.get(model_id)
    if model is None:
        raise KeyError(f"Unknown model: {model_id}")
    for op in model.gemm_ops:
        if op.id == gemm_id:
            return op
    raise KeyError(f"Unknown GEMM op {gemm_id} for model {model_id}")

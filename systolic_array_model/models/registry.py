"""Model GEMM operation registry (Llama3-8B decode, tp=1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from systolic_array_model.models.benchmarks import get_benchmarks
from systolic_array_model.hierarchy import (
    DIE_PPU_COUNT,
    OS_K_MAX,
    OS_M_MAX,
    OS_N_MAX,
    OS_N_WAVE_MAX,
    one_head_per_ppu_layout,
    os_n_wave_count,
    os_ppu_n_wave_count,
    ppus_needed_for_n,
)


@dataclass(frozen=True)
class GemmOp:
    id: str
    label: str
    m: int
    k: int
    n: int
    batch: int = 1
    note: str = ""

    def fits_die(self) -> bool:
        if self.m < 1 or self.m > OS_M_MAX:
            return False
        if self.k < 1 or self.k > OS_K_MAX:
            return False
        if self.batch > 1:
            try:
                one_head_per_ppu_layout(self.batch)
            except ValueError:
                return False
        return True

    def wave_count(self) -> int:
        if self.batch <= 1:
            return os_n_wave_count(self.n)
        return os_ppu_n_wave_count(self.n)

    def ppus_needed(self) -> int:
        """Peak PPU usage across the busiest wave."""
        if self.batch <= 1:
            return ppus_needed_for_n(self.n)
        return min(self.batch, DIE_PPU_COUNT)

    def heads_per_wave(self) -> int:
        if self.batch <= 1:
            return 1
        heads, _ = one_head_per_ppu_layout(self.batch)
        return heads

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "batch": self.batch,
            "heads_per_wave": self.heads_per_wave(),
            "note": self.note,
            "ppus_needed": self.ppus_needed(),
            "wave_count": self.wave_count(),
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
_ATTN_SEQ = 2048


LLAMA3_8B = ModelSpec(
    id="llama3-8b",
    label="Llama3-8B",
    description="Decode 单 token 步 (batch=1, seq=1, tp=1)",
    gemm_ops=[
        GemmOp("qo_proj", "Q/O_Proj", _DECODE_M, _HIDDEN, _HIDDEN, note="Q_Proj · O_Proj"),
        GemmOp("kv_proj", "K/V_Proj", _DECODE_M, _HIDDEN, _KV_HEADS * _HEAD_DIM, note="K_Proj · V_Proj"),
        GemmOp(
            "qkt_bmm",
            "Q @ K.T",
            _DECODE_M,
            _HEAD_DIM,
            _ATTN_SEQ,
            batch=_HEADS,
            note="BMM · Attention-1 · seq_len=2048 · 32 head 均分 32 PPU",
        ),
        GemmOp(
            "pv_bmm",
            "P @ V",
            _DECODE_M,
            _ATTN_SEQ,
            _HEAD_DIM,
            batch=_HEADS,
            note="BMM · Attention-2 · seq_len=2048 · 32 head 均分 32 PPU",
        ),
        GemmOp("gate_up_proj", "Gate/Up_Proj", _DECODE_M, _HIDDEN, _INTERMEDIATE, note="Gate_Proj · Up_Proj"),
        GemmOp("down_proj", "Down_Proj", _DECODE_M, _INTERMEDIATE, _HIDDEN),
        GemmOp("lm_head", "LM_Head", _DECODE_M, _HIDDEN, _VOCAB),
    ],
)

_MODELS: dict[str, ModelSpec] = {
    LLAMA3_8B.id: LLAMA3_8B,
}


def _benchmark_for_model(model_id: str) -> tuple[str, dict[str, int]]:
    for entry in get_benchmarks(model_id=model_id):
        label = str(entry.get("label", ""))
        cycles = entry.get("gemm_cycles", {})
        return label, {str(k): int(v) for k, v in cycles.items()}
    return "", {}


def _model_to_catalog_dict(spec: ModelSpec) -> dict:
    ref_label, ref_cycles = _benchmark_for_model(spec.id)
    gemm_ops: list[dict] = []
    for op in spec.gemm_ops:
        op_dict = op.to_dict()
        ref = ref_cycles.get(op.id)
        if ref is not None:
            op_dict["ref_cycles"] = ref
            op_dict["ref_label"] = ref_label
        gemm_ops.append(op_dict)
    return {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "gemm_ops": gemm_ops,
    }


def get_models_catalog() -> dict:
    return {
        "models": [_model_to_catalog_dict(spec) for spec in _MODELS.values()],
        "default_model_id": LLAMA3_8B.id,
        "default_gemm_id": "qo_proj",
        "hardware": {
            "die_ppu_count": DIE_PPU_COUNT,
            "os_m_max": OS_M_MAX,
            "os_k_max": OS_K_MAX,
            "os_n_max": OS_N_MAX,
            "os_n_wave_max": OS_N_WAVE_MAX,
            "memory": "sram_only",
        },
        "benchmarks": get_benchmarks(),
    }


def get_gemm_op(model_id: str, gemm_id: str) -> GemmOp:
    model = _MODELS.get(model_id)
    if model is None:
        raise KeyError(f"Unknown model: {model_id}")
    for op in model.gemm_ops:
        if op.id == gemm_id:
            return op
    raise KeyError(f"Unknown GEMM op {gemm_id} for model {model_id}")

import csv
import os

from config.model_config import ModelConfig
from hardware.gpu import GPU

# Matches kernel_benchmark/collective_communication.py and bench_data/comm CSV "op" column.
COLLECTIVE_OPS = ("all_reduce", "all_gather", "reduce_scatter")

# (device_type, tp_size) -> CSV rows
COMM_BENCH_DATA: dict[tuple[str, int], list[dict[str, str]]] = {}


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_comm_bench(device_type: str, tp_size: int) -> list[dict[str, str]]:
    """Load bench_data/comm/<device>/ws<N>_data.csv (all columns, one dict per row)."""
    key = (device_type.lower(), tp_size)
    if key in COMM_BENCH_DATA:
        return COMM_BENCH_DATA[key]

    path = os.path.join(
        repo_root(),
        "bench_data",
        "comm",
        device_type.lower(),
        f"ws{tp_size}_data.csv",
    )
    if not os.path.exists(path):
        rows: list[dict[str, str]] = []
    else:
        with open(path, "r") as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if int(row["world_size"]) == tp_size
            ]

    COMM_BENCH_DATA[key] = rows
    return rows


def find_nearest_comm_bench_row(
    device_type: str,
    tp_size: int,
    op: str,
    per_rank_bytes: int,
) -> dict[str, str] | None:
    """Bench row with num_bytes>=per_rank_bytes and smallest excess."""
    if op not in COLLECTIVE_OPS:
        raise ValueError(f"op must be one of {COLLECTIVE_OPS}, got {op!r}")

    best_row = None
    best_excess = None
    best_latency_us = None
    for row in load_comm_bench(device_type, tp_size):
        if row["op"] != op:
            continue
        bench_bytes = int(row["num_bytes"])
        if bench_bytes < per_rank_bytes:
            continue
        excess = bench_bytes - per_rank_bytes
        latency_us = float(row["latency_us"])
        if best_excess is None or excess < best_excess or (
            excess == best_excess
            and (best_latency_us is None or latency_us < best_latency_us)
        ):
            best_excess = excess
            best_latency_us = latency_us
            best_row = row
    return best_row


def collective_transfer_bytes(op: str, per_rank_bytes: int, tp_size: int) -> float:
    """Bus traffic bytes (nccl-tests busbw model, see collective_communication.nccl_bandwidth)."""
    n = tp_size
    if op == "all_reduce":
        return per_rank_bytes * 2 * (n - 1) / n
    if op == "all_gather":
        return per_rank_bytes * (n - 1)
    if op == "reduce_scatter":
        return per_rank_bytes * (n - 1) / n
    raise ValueError(f"unsupported op: {op}")


def estimate_collective_latency_us(
    peak_bw: float,
    op: str,
    tp_size: int,
    per_rank_bytes: int,
    bench_row: dict[str, str],
) -> float:
    """Scale actual transfer by busbw_ratio from a nearby bench point vs peak link bw."""
    busbw_ratio = float(bench_row["busbw_ratio"])
    if busbw_ratio <= 0:
        busbw_ratio = float(bench_row["busbw_gbps"]) / peak_bw

    transfer_bytes = collective_transfer_bytes(op, per_rank_bytes, tp_size)
    # busbw_ratio = measured_bus_bw / peak_bw (see collective_communication.py)
    effective_bw = peak_bw * busbw_ratio
    return transfer_bytes / effective_bw / 1e9 * 1e6


class Comm:
    def __init__(
        self,
        config: ModelConfig,
        gpu: GPU,
        tp_size: int = 1,
        device_type: str = "A100",
    ):
        self.config = config
        self.gpu = gpu
        self.tp_size = tp_size
        self.device_type = device_type

    def collective_latency_s(self, op: str, per_rank_bytes: int) -> float:
        """Latency (seconds) for one collective; per_rank_bytes is send buffer per GPU."""
        if self.tp_size <= 1:
            return 0.0

        peak_bw = self.gpu.nvlink_bw
        bench_row = find_nearest_comm_bench_row(
            self.device_type, self.tp_size, op, per_rank_bytes
        )
        if bench_row is not None:
            latency_us = estimate_collective_latency_us(
                peak_bw, op, self.tp_size, per_rank_bytes, bench_row
            )
            return latency_us * 1e-6

        print(
            f"Warning: no comm bench row for op={op}, ws={self.tp_size}, "
            f"per_rank_bytes={per_rank_bytes} "
            f"(device={self.device_type}); use peak link bandwidth fallback"
        )
        transfer_bytes = collective_transfer_bytes(op, per_rank_bytes, self.tp_size)
        return transfer_bytes / (1024**3) / peak_bw

    def all_reduce_latency_s(self, per_rank_bytes: int) -> float:
        return self.collective_latency_s("all_reduce", per_rank_bytes)
    
    def all_gather_latency_s(self, per_rank_bytes: int) -> float:
        return self.collective_latency_s("all_gather", per_rank_bytes)

    def prefill_comm(self, batch_size: int, seq_len: int):
        """Return (comm after O_proj, comm after FFN down_proj) in seconds."""
        if self.tp_size <= 1:
            return 0.0, 0.0
        per_rank_bytes = batch_size * seq_len * self.config.hidden_size * 2
        return (
            self.all_reduce_latency_s(per_rank_bytes),
            self.all_reduce_latency_s(per_rank_bytes),
        )

    def decode_comm(self, batch_size: int, seq_len: int = 1):
        if self.tp_size <= 1:
            return 0.0, 0.0
        per_rank_bytes = batch_size * seq_len * self.config.hidden_size * 2
        return (
            self.all_reduce_latency_s(per_rank_bytes),
            self.all_reduce_latency_s(per_rank_bytes),
        )

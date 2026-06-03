# Inter-GPU collective communication bandwidth benchmark (NCCL via torch.distributed).
#
# Launch example (8 GPUs on one node, use all 8):
#   torchrun --nproc_per_node=8 kernel_benchmark/collective_communication.py \
#       --device-type A100 --dtype fp16
#
# Use only physical GPU 4,5,6,7 on an 8xA100 node (4 processes):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
#       kernel_benchmark/collective_communication.py --device-type A100 --dtype fp16
#
import argparse
import os
import random
import sys
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from hardware.gpu import gpu_map  # noqa: E402

# Token count M in activation [M, H] (decode: M=bs, prefill: M=bs*seq_len).
# all_reduce: in-place, no extra output buffer.
DEFAULT_AR_M_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]

DEFAULT_AR_H_SIZES = [
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

# all_gather / reduce_scatter need extra output buffers; use smaller sweeps.
# all_gather per rank: input m*h + output m*h*world_size
DEFAULT_AG_RS_M_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

DEFAULT_AG_RS_H_SIZES = [
    2048,
    4096,
    8192,
    16384,
    32768,
]

DEFAULT_OPS = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
]

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def bytes_per_elem(dtype: torch.dtype) -> int:
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]


def parse_sizes(sizes_arg: Optional[str], default: list[int]) -> list[int]:
    if sizes_arg:
        return [int(x) for x in sizes_arg.split(",")]
    return default


def parse_ops(ops_arg: Optional[str]) -> list[str]:
    if ops_arg:
        ops = [x.strip() for x in ops_arg.split(",")]
    else:
        ops = list(DEFAULT_OPS)
    valid = set(DEFAULT_OPS)
    for op in ops:
        if op not in valid:
            raise ValueError(f"Unknown op {op!r}, choose from {sorted(valid)}")
    return ops


def get_mh_sizes_for_op(
    op: str,
    ar_m: list[int],
    ar_h: list[int],
    ag_rs_m: list[int],
    ag_rs_h: list[int],
) -> tuple[list[int], list[int]]:
    if op == "all_reduce":
        return ar_m, ar_h
    return ag_rs_m, ag_rs_h


def init_distributed() -> tuple[int, int, int]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if "RANK" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun, e.g. "
            "torchrun --nproc_per_node=8 kernel_benchmark/collective_communication.py"
        )
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def time_collective(
    op: str,
    tensor: torch.Tensor,
    out: torch.Tensor | None,
    warmup: int,
    repeats: int,
) -> float:
    """Return mean latency in seconds for one collective call."""
    for _ in range(warmup):
        _run_collective(op, tensor, out)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _run_collective(op, tensor, out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats / 1e3  # ms -> s


def _run_collective(op: str, tensor: torch.Tensor, out: torch.Tensor | None) -> None:
    if op == "all_reduce":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "all_gather":
        dist.all_gather_into_tensor(out, tensor)
    elif op == "reduce_scatter":
        dist.reduce_scatter_tensor(out, tensor, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unknown op: {op}")


def nccl_bandwidth(
    op: str, world_size: int, count_per_rank: int, elem_bytes: int, latency_s: float
) -> tuple[float, float]:
    """Return (algorithm_GB/s, bus_GB/s) following nccl-tests conventions."""
    if latency_s <= 0:
        return 0.0, 0.0
    n = world_size

    if op == "all_reduce":
        algbw = count_per_rank * elem_bytes / latency_s / 1e9
        busbw = algbw * 2 * (n - 1) / n
    elif op == "all_gather":
        algbw = n * count_per_rank * elem_bytes / latency_s / 1e9
        busbw = algbw * (n - 1) / n
    elif op == "reduce_scatter":
        algbw = count_per_rank * elem_bytes / latency_s / 1e9
        busbw = algbw * (n - 1) / n
    else:
        raise ValueError(op)

    return algbw, busbw


def test_collective(
    op: str,
    m: int,
    h: int,
    dtype: torch.dtype,
    world_size: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, int]:
    """Benchmark one (op, m, h) case. Tensor per rank: [m, h] flattened."""
    elem_bytes = bytes_per_elem(dtype)
    count_per_rank = m * h
    num_bytes = count_per_rank * elem_bytes

    if op == "reduce_scatter" and count_per_rank % world_size != 0:
        raise ValueError(
            f"reduce_scatter requires m*h divisible by world_size, "
            f"got m={m}, h={h}, world_size={world_size}"
        )

    tensor = torch.randn(count_per_rank, device="cuda", dtype=dtype)
    out = None
    if op == "all_gather":
        out = torch.empty(
            count_per_rank * world_size, device="cuda", dtype=dtype
        )
    elif op == "reduce_scatter":
        out = torch.empty(count_per_rank // world_size, device="cuda", dtype=dtype)

    latency_s = time_collective(op, tensor, out, warmup, repeats)
    algbw, busbw = nccl_bandwidth(op, world_size, count_per_rank, elem_bytes, latency_s)
    latency_us = latency_s * 1e6

    if dist.get_rank() == 0:
        dtype_name = str(dtype).split(".")[-1]
        print(
            f" > {op:14} (m={m:6}, h={h:5}, {dtype_name}, ws={world_size}): "
            f"bytes/rank={num_bytes/1024/1024:8.2f} MB | "
            f"{latency_us:8.1f} us | "
            f"algbw {algbw:7.2f} GB/s | busbw {busbw:7.2f} GB/s"
        )
    return latency_us, algbw, busbw, num_bytes


def main(args) -> None:
    rank, world_size, _ = init_distributed()

    dtype = DTYPE_MAP[args.dtype]
    ops = parse_ops(args.ops)
    gpu = gpu_map[args.device_type]
    peak_nvlink = gpu.nvlink_bw

    ar_m = parse_sizes(args.m_sizes, DEFAULT_AR_M_SIZES)
    ar_h = parse_sizes(args.h_sizes, DEFAULT_AR_H_SIZES)
    ag_rs_m = parse_sizes(args.ag_rs_m_sizes, DEFAULT_AG_RS_M_SIZES)
    ag_rs_h = parse_sizes(args.ag_rs_h_sizes, DEFAULT_AG_RS_H_SIZES)

    if rank == 0:
        print("collective_communication benchmark (torch.distributed NCCL)")
        print(f" > device: {torch.cuda.get_device_name()}")
        print(f" > world_size: {world_size}")
        print(f" > dtype: {args.dtype}")
        print(f" > ops: {ops}")
        print(f" > reference nvlink_bw (hardware/gpu.py): {peak_nvlink:.1f} GB/s")
        print(f" > all_reduce M/H: {ar_m} / {ar_h}")
        print(f" > all_gather & reduce_scatter M/H: {ag_rs_m} / {ag_rs_h}")
        print(f" > warmup={args.warmup}, repeats={args.repeats}\n")

    results = []
    for op in ops:
        m_sizes, h_sizes = get_mh_sizes_for_op(op, ar_m, ar_h, ag_rs_m, ag_rs_h)
        if rank == 0:
            print(f"--- {op}: {len(m_sizes) * len(h_sizes)} cases ---")
        for h in h_sizes:
            for m in m_sizes:
                if op == "reduce_scatter" and (m * h) % world_size != 0:
                    if rank == 0:
                        print(
                            f" ! skip reduce_scatter m={m}, h={h}: "
                            f"m*h not divisible by world_size={world_size}"
                        )
                    continue
                latency_us, algbw, busbw, num_bytes = test_collective(
                    op, m, h, dtype, world_size, args.warmup, args.repeats
                )
                if rank == 0:
                    results.append(
                        {
                            "op": op,
                            "world_size": world_size,
                            "m": m,
                            "h": h,
                            "num_bytes": num_bytes,
                            "latency_us": round(latency_us, 3),
                            "algbw_gbps": round(algbw, 3),
                            "busbw_gbps": round(busbw, 3),
                            "busbw_ratio": round(busbw / peak_nvlink, 6),
                        }
                    )

    if rank == 0:
        df = pd.DataFrame(results)
        out_path = args.output
        if out_path is None:
            out_dir = os.path.join(
                parent_dir,
                "bench_data",
                "comm",
                args.device_type.lower(),
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"ws{world_size}_data.csv")

        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(results)} rows to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Benchmark NCCL collective bandwidth (all_reduce / all_gather / reduce_scatter)"
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated ops; default: all_reduce,all_gather,reduce_scatter",
    )
    parser.add_argument(
        "--m-sizes",
        type=str,
        default=None,
        help="Comma-separated M for all_reduce only; default DEFAULT_AR_M_SIZES",
    )
    parser.add_argument(
        "--h-sizes",
        type=str,
        default=None,
        help="Comma-separated H for all_reduce only; default DEFAULT_AR_H_SIZES",
    )
    parser.add_argument(
        "--ag-rs-m-sizes",
        type=str,
        default=None,
        help="Comma-separated M for all_gather & reduce_scatter; default DEFAULT_AG_RS_M_SIZES",
    )
    parser.add_argument(
        "--ag-rs-h-sizes",
        type=str,
        default=None,
        help="Comma-separated H for all_gather & reduce_scatter; default DEFAULT_AG_RS_H_SIZES",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=list(DTYPE_MAP.keys()),
        help="Tensor dtype",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="A100",
        choices=list(gpu_map.keys()),
        help="Used for bench_data output path and nvlink reference",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Timed iterations")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path (default: bench_data/comm/<device>/ws<N>_data.csv)",
    )
    main(parser.parse_args())

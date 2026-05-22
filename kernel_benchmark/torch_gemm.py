# GEMM throughput benchmark via torch.nn.Linear (no DeepGEMM).
import argparse
import os
import random
import sys
from typing import Optional

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from hardware.gpu import gpu_map  # noqa: E402

DEFAULT_M_SIZES = [
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
]

# Edit these lists for your target model / layer shapes.
DEFAULT_K_SIZES = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    14336,
]

DEFAULT_N_SIZES = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    14336,
    128 * 1024,
]

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def benchmark_forward(fn, *inputs, repeats=20, **kwinputs):
    t = benchmark.Timer(
        stmt="fn(*inputs, **kwinputs)",
        globals={"fn": fn, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    return t.timeit(repeats)


def time_us(fn, *args, **kwargs):
    return benchmark_forward(fn, *args, **kwargs).mean * 1e6


def bytes_per_elem(dtype: torch.dtype) -> int:
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]


def test_gemm(
    m: int, k: int, n: int, dtype: torch.dtype, warmup: int
) -> tuple[float, float, float]:
    """Run y = x @ W^T with torch.nn.Linear(in_features=k, out_features=n)."""
    linear = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    def run():
        linear(x)

    # correctness
    # with torch.no_grad():
    #     out = linear(x)
    #     ref = x.float() @ linear.weight.float().T
    #     diff = (out.float() - ref).abs().max().item()
    #     assert diff < 1e-1 if dtype == torch.bfloat16 else diff < 1e-2, (
    #         f"max diff {diff:.6f} for m={m}, k={k}, n={n}"
    #     )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    latency_us = time_us(run)
    torch.cuda.synchronize()

    t_s = latency_us / 1e6
    tflops = 2 * m * n * k / t_s / 1e12
    bpe = bytes_per_elem(dtype)
    bytes_moved = (m * k + k * n + m * n) * bpe
    gbps = bytes_moved / t_s / 1e9

    dtype_name = str(dtype).split(".")[-1]
    print(
        f" > Perf (m={m:6}, k={k:5}, n={n:5}, {dtype_name}): "
        f"{latency_us:8.3f} us | {tflops:6.3f} TFLOPS | {gbps:6.3f} GB/s"
    )
    return latency_us, tflops, gbps


def parse_sizes(sizes_arg: Optional[str], default: list[int]) -> list[int]:
    if sizes_arg:
        return [int(x) for x in sizes_arg.split(",")]
    return default


def peak_tflops(
    device_type: str, dtype_name: str, gpu_tflops: Optional[float]
) -> float:
    if gpu_tflops is not None:
        return gpu_tflops
    gpu = gpu_map[device_type]
    if dtype_name == "fp32":
        return gpu.fp16_tflops / 2
    return gpu.fp16_tflops


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch_gemm benchmark")

    dtype = DTYPE_MAP[args.dtype]
    peak = peak_tflops(args.device_type, args.dtype, args.gpu_tflops)
    m_sizes = parse_sizes(args.m_sizes, DEFAULT_M_SIZES)
    k_sizes = parse_sizes(args.k_sizes, DEFAULT_K_SIZES)
    n_sizes = parse_sizes(args.n_sizes, DEFAULT_N_SIZES)

    print("torch_gemm benchmark (torch.nn.Linear)")
    print(f" > device: {torch.cuda.get_device_name()}")
    print(f" > dtype: {args.dtype}, peak TFLOPS (for MFU): {peak}")
    print(f" > m sizes ({len(m_sizes)}): {m_sizes}")
    print(f" > k sizes ({len(k_sizes)}): {k_sizes}")
    print(f" > n sizes ({len(n_sizes)}): {n_sizes}")
    print(
        f" > total cases: {len(m_sizes) * len(k_sizes) * len(n_sizes)}, "
        f"warmup={args.warmup}\n"
    )

    results = []
    for k in k_sizes:
        for n in n_sizes:
            for m in m_sizes:
                latency_us, tflops, gbps = test_gemm(m, k, n, dtype, args.warmup)
                results.append(
                    {
                        "m": m,
                        "k": k,
                        "n": n,
                        "latency_us": round(latency_us, 3),
                        "mfu": round(tflops / peak, 6),
                        "tflops": round(tflops, 3),
                        "gbps": round(gbps, 3),
                    }
                )

    df = pd.DataFrame(results)
    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(
            parent_dir, "bench_data", "gemm", args.device_type.lower()
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.csv")

    # bench_data/gemm/*.csv: m,k,n,latency_us,mfu
    df[["m", "k", "n", "latency_us", "mfu"]].to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Measure GEMM throughput with torch.nn.Linear"
    )
    parser.add_argument(
        "--k-sizes",
        type=str,
        default=None,
        help="Comma-separated in_features (k); default uses DEFAULT_K_SIZES in script",
    )
    parser.add_argument(
        "--n-sizes",
        type=str,
        default=None,
        help="Comma-separated out_features (n); default uses DEFAULT_N_SIZES in script",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=list(DTYPE_MAP.keys()),
        help="GEMM dtype",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="A100",
        choices=list(gpu_map.keys()),
        help="Used for default peak TFLOPS and bench_data output path",
    )
    parser.add_argument(
        "--gpu-tflops",
        type=float,
        default=None,
        help="Peak TFLOPS for MFU; default from hardware/gpu.py fp16_tflops",
    )
    parser.add_argument(
        "--m-sizes",
        type=str,
        default=None,
        help="Comma-separated batch dims m; default uses DEFAULT_M_SIZES in script",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path (default: bench_data/gemm/<device>/data.csv)",
    )
    main(parser.parse_args())

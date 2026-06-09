"""CLI for systolic array dataflow simulation."""

from __future__ import annotations

import argparse
import json
import sys

from systolic_array.simulator import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Systolic array GEMM dataflow simulator")
    parser.add_argument("--m", type=int, default=4, help="Matrix A rows (M)")
    parser.add_argument("--k", type=int, default=4, help="Inner dimension K")
    parser.add_argument("--n", type=int, default=4, help="Matrix B cols (N)")
    parser.add_argument("--rows", type=int, default=16, help="Array rows")
    parser.add_argument("--cols", type=int, default=16, help="Array cols")
    parser.add_argument("--dataflow", choices=["output_stationary", "weight_stationary"], default="output_stationary")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args()

    from systolic_array.types import DataflowType

    result = run_simulation(
        args.m, args.k, args.n,
        rows=args.rows, cols=args.cols,
        dataflow=DataflowType(args.dataflow),
    )

    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"Array: {args.rows}x{args.cols}")
        print(f"GEMM: A[{args.m}x{args.k}] x B[{args.k}x{args.n}]")
        print(f"Total cycles: {result.total_cycles}")
        print(f"Snapshots: {len(result.snapshots)}")


if __name__ == "__main__":
    main()

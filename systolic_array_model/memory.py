"""SRAM load/store model for dataflow visualization (no DDR hierarchy)."""

from __future__ import annotations

from systolic_array_model.types import DataflowType, SimPhase

FP16_BYTES = 2
DEFAULT_DTYPE = "fp16"


def empty_memory_access(
    *,
    bytes_per_elem: int = FP16_BYTES,
    dtype: str = DEFAULT_DTYPE,
) -> dict[str, int | str | dict[str, int]]:
    return {
        "dtype": dtype,
        "bytes_per_elem": bytes_per_elem,
        "read_bytes": 0,
        "write_bytes": 0,
        "read_elems": 0,
        "write_elems": 0,
        "read_breakdown": {"weight": 0, "activation": 0},
        "write_breakdown": {"output": 0},
    }


def peak_memory_access(
    memories: list[dict[str, int | str | dict[str, int]]],
) -> dict[str, int | str | dict[str, int]]:
    """Peak bandwidth = max over cycles of (read_bytes + write_bytes)."""
    if not memories:
        return {
            "dtype": DEFAULT_DTYPE,
            "bytes_per_elem": FP16_BYTES,
            "peak_bytes": 0,
            "peak_cycle": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_breakdown": {"weight": 0, "activation": 0},
            "write_breakdown": {"output": 0},
        }

    dtype = str(memories[0].get("dtype", DEFAULT_DTYPE))
    bytes_per_elem = int(memories[0].get("bytes_per_elem", FP16_BYTES))

    peak_bytes = -1
    peak_cycle = 0
    peak_mem = memories[0]

    for cycle, mem in enumerate(memories):
        total_bytes = int(mem.get("read_bytes", 0)) + int(mem.get("write_bytes", 0))
        if total_bytes > peak_bytes:
            peak_bytes = total_bytes
            peak_cycle = cycle
            peak_mem = mem

    rb = peak_mem.get("read_breakdown", {})
    wb = peak_mem.get("write_breakdown", {})
    if not isinstance(rb, dict):
        rb = {}
    if not isinstance(wb, dict):
        wb = {}

    return {
        "dtype": dtype,
        "bytes_per_elem": bytes_per_elem,
        "peak_bytes": peak_bytes,
        "peak_cycle": peak_cycle,
        "read_bytes": int(peak_mem.get("read_bytes", 0)),
        "write_bytes": int(peak_mem.get("write_bytes", 0)),
        "read_breakdown": {
            "weight": int(rb.get("weight", 0)),
            "activation": int(rb.get("activation", 0)),
        },
        "write_breakdown": {"output": int(wb.get("output", 0))},
    }


def merge_memory_access(
    left: dict[str, int | str | dict[str, int]],
    right: dict[str, int | str | dict[str, int]],
) -> dict[str, int | str | dict[str, int]]:
    """Sum per-cycle memory stats (e.g. across cores in one PPU)."""
    l_rb = left.get("read_breakdown", {})
    r_rb = right.get("read_breakdown", {})
    l_wb = left.get("write_breakdown", {})
    r_wb = right.get("write_breakdown", {})
    if not isinstance(l_rb, dict):
        l_rb = {}
    if not isinstance(r_rb, dict):
        r_rb = {}
    if not isinstance(l_wb, dict):
        l_wb = {}
    if not isinstance(r_wb, dict):
        r_wb = {}

    read_bytes = int(left.get("read_bytes", 0)) + int(right.get("read_bytes", 0))
    write_bytes = int(left.get("write_bytes", 0)) + int(right.get("write_bytes", 0))

    return {
        "dtype": str(left.get("dtype", right.get("dtype", DEFAULT_DTYPE))),
        "bytes_per_elem": int(left.get("bytes_per_elem", right.get("bytes_per_elem", FP16_BYTES))),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "read_elems": int(left.get("read_elems", 0)) + int(right.get("read_elems", 0)),
        "write_elems": int(left.get("write_elems", 0)) + int(right.get("write_elems", 0)),
        "read_breakdown": {
            "weight": int(l_rb.get("weight", 0)) + int(r_rb.get("weight", 0)),
            "activation": int(l_rb.get("activation", 0)) + int(r_rb.get("activation", 0)),
        },
        "write_breakdown": {
            "output": int(l_wb.get("output", 0)) + int(r_wb.get("output", 0)),
        },
    }


def _spread_read_elems(total: int, load_cycles: int, cycle_idx: int) -> int:
    if load_cycles <= 0:
        return total
    base, extra = divmod(total, load_cycles)
    return base + (1 if cycle_idx < extra else 0)


def os_writeback_count(m: int, k: int, n: int, local_cycle: int) -> int:
    """PE(m,c) writes when its last MAC completes at local_cycle = m + c + K - 1."""
    if local_cycle < 0 or k <= 0:
        return 0
    done = local_cycle - k + 1
    if done < 0 or done > (m - 1) + (n - 1):
        return 0
    count = 0
    for r in range(m):
        c = done - r
        if 0 <= c < n:
            count += 1
    return count


def os_memory_at_cycle(
    m: int,
    k: int,
    n: int,
    local_cycle: int,
    *,
    m0: int = 0,
    k0: int = 0,
    n0: int = 0,
) -> dict[str, int | str | dict[str, int]]:
    """Analytical per-cycle OS memory traffic (no micro re-simulation)."""
    left_labels: list[str | None] = []
    for r in range(m):
        k_idx = local_cycle - r
        left_labels.append(
            f"A[{r + m0},{k_idx + k0}]" if 0 <= k_idx < k else None
        )
    top_labels: list[str | None] = []
    for c in range(n):
        k_idx = local_cycle - c
        top_labels.append(
            f"W[{k_idx + k0},{c + n0}]" if 0 <= k_idx < k else None
        )
    return compute_memory_access(
        phase=SimPhase.COMPUTE,
        dataflow=DataflowType.OUTPUT_STATIONARY,
        m=m,
        k=k,
        n=n,
        left_labels=left_labels,
        top_labels=top_labels,
        local_cycle=local_cycle,
    )


def compute_memory_access(
    *,
    phase: SimPhase,
    dataflow: DataflowType,
    m: int,
    k: int,
    n: int,
    left_labels: list[str | None],
    top_labels: list[str | None] | None = None,
    bottom_labels: list[str | None] | None = None,
    weight_load_cycles: int = 1,
    weight_load_idx: int = 0,
    local_cycle: int = -1,
    bytes_per_elem: int = FP16_BYTES,
    dtype: str = DEFAULT_DTYPE,
) -> dict[str, int | str | dict[str, int]]:
    """Estimate external memory traffic for one cycle."""
    read_weight = 0
    read_activation = 0
    write_output = 0

    if dataflow == DataflowType.WEIGHT_STATIONARY:
        if phase == SimPhase.WEIGHT_LOAD:
            read_weight = _spread_read_elems(k * n, weight_load_cycles, weight_load_idx)
        elif phase == SimPhase.COMPUTE:
            read_activation = sum(1 for label in left_labels if label)
            if bottom_labels:
                write_output = sum(1 for label in bottom_labels if label)
    elif dataflow == DataflowType.OUTPUT_STATIONARY:
        if phase == SimPhase.COMPUTE:
            read_activation = sum(1 for label in left_labels if label)
            if top_labels:
                read_weight = sum(1 for label in top_labels if label)
            write_output = os_writeback_count(m, k, n, local_cycle)

    read_elems = read_weight + read_activation
    write_elems = write_output

    return {
        "dtype": dtype,
        "bytes_per_elem": bytes_per_elem,
        "read_bytes": read_elems * bytes_per_elem,
        "write_bytes": write_elems * bytes_per_elem,
        "read_elems": read_elems,
        "write_elems": write_elems,
        "read_breakdown": {
            "weight": read_weight * bytes_per_elem,
            "activation": read_activation * bytes_per_elem,
        },
        "write_breakdown": {
            "output": write_output * bytes_per_elem,
        },
    }

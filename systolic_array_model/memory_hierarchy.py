"""LPDDR ↔ PPU SRAM hierarchy: A full resident, W K-tiled double-buffer prefetch, C writeback."""

from __future__ import annotations

import math
from typing import Any

from systolic_array_model.cycles import compute_os_cycle_count
from systolic_array_model.memory import FP16_BYTES

DEFAULT_PPU_SRAM_KB = 256
DEFAULT_LPDDR_GBPS = 256.0
DEFAULT_CLOCK_GHZ = 1.0


def lpddr_bytes_per_cycle(bandwidth_gbps: float, clock_ghz: float) -> float:
    """Convert peak LPDDR bandwidth (GB/s) to bytes per logic clock cycle."""
    if clock_ghz <= 0:
        return 0.0
    return bandwidth_gbps * 1e9 / (clock_ghz * 1e9)


def ppu_sram_bytes(ppu_sram_kb: int) -> int:
    return ppu_sram_kb * 1024


def w_fits_in_sram(local_m: int, local_k: int, local_n: int, *, ppu_sram_kb: int) -> bool:
    """True when full A and W tiles fit in PPU SRAM (no K streaming required)."""
    sram = ppu_sram_bytes(ppu_sram_kb)
    a_bytes = local_m * local_k * FP16_BYTES
    w_bytes = local_k * local_n * FP16_BYTES
    return a_bytes + w_bytes <= sram


def compute_k_chunk_elems(
    local_m: int,
    local_k: int,
    local_n: int,
    *,
    ppu_sram_kb: int,
    double_buffer: bool = True,
) -> int:
    """Max K columns per W prefetch buffer after full A resident."""
    sram = ppu_sram_bytes(ppu_sram_kb)
    a_bytes = local_m * local_k * FP16_BYTES
    if a_bytes > sram:
        raise ValueError(
            f"A tile ({local_m}×{local_k}) needs {a_bytes} B but PPU SRAM is {sram} B"
        )
    if local_n <= 0:
        return local_k
    w_bytes = local_k * local_n * FP16_BYTES
    remain = sram - a_bytes
    w_buf_one = remain // 2 if double_buffer else remain
    if w_bytes <= w_buf_one:
        return local_k
    k_chunk = w_buf_one // (local_n * FP16_BYTES)
    return max(1, min(k_chunk, local_k))


def split_k_chunks(local_k: int, k_chunk: int) -> list[int]:
    chunks: list[int] = []
    k0 = 0
    while k0 < local_k:
        ki = min(k_chunk, local_k - k0)
        chunks.append(ki)
        k0 += ki
    return chunks


def cycles_for_bytes(total_bytes: int, bytes_per_cycle: float) -> int:
    if total_bytes <= 0:
        return 0
    if bytes_per_cycle <= 0:
        return 0
    return max(1, math.ceil(total_bytes / bytes_per_cycle))


def double_buffer_pipeline_cycles(
    prefetch_cycles: list[int],
    compute_cycles: list[int],
) -> int:
    """One buffer prefetches while the other computes (after the first fill)."""
    n = len(prefetch_cycles)
    if n == 0:
        return 0
    if n != len(compute_cycles):
        raise ValueError("prefetch and compute chunk counts must match")
    if n == 1:
        return prefetch_cycles[0] + compute_cycles[0]
    total = prefetch_cycles[0]
    for i in range(n - 1):
        total += max(compute_cycles[i], prefetch_cycles[i + 1])
    total += compute_cycles[n - 1]
    return total


def _ppu_summaries(wave_tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ppu: dict[int, list[dict[str, Any]]] = {}
    for tile in wave_tiles:
        ppu = int(tile["ppu_index"])
        by_ppu.setdefault(ppu, []).append(tile)

    summaries: list[dict[str, Any]] = []
    for ppu_index in sorted(by_ppu):
        tiles = by_ppu[ppu_index]
        local_m = int(tiles[0]["local_m"])
        local_k = int(tiles[0]["local_k"])
        local_n = sum(int(t["local_n"]) for t in tiles)
        summaries.append(
            {
                "ppu_index": ppu_index,
                "local_m": local_m,
                "local_k": local_k,
                "local_n": local_n,
            }
        )
    return summaries


def _wave_compute_only_cycles(wave_tiles: list[dict[str, Any]]) -> int:
    if not wave_tiles:
        return 0
    return max(
        compute_os_cycle_count(
            int(t["local_m"]),
            int(t["local_k"]),
            int(t["local_n"]),
        )
        for t in wave_tiles
    )


def logic_die_lpddr_schedule(
    tile_plan: dict[str, Any],
    dims: dict[str, int],
    *,
    compute_only_cycles: int,
    ppu_sram_kb: int = DEFAULT_PPU_SRAM_KB,
    lpddr_bandwidth_gbps: float = DEFAULT_LPDDR_GBPS,
    clock_ghz: float = DEFAULT_CLOCK_GHZ,
) -> dict[str, Any]:
    """Double-buffer W prefetch from LPDDR; A broadcast once per wave; C writeback after last K tile."""
    lpddr_bpc = lpddr_bytes_per_cycle(lpddr_bandwidth_gbps, clock_ghz)
    waves_meta = tile_plan.get("waves") or [{"wave_index": 0}]
    m = int(dims.get("m", 1))
    k = int(dims.get("k", 1))

    wave_stats: list[dict[str, Any]] = []
    total_pipeline = 0
    total_compute_only = 0

    for wave in waves_meta:
        wave_idx = int(wave.get("wave_index", 0))
        wave_tiles = [
            t
            for t in tile_plan.get("tiles", [])
            if int(t.get("wave_index", 0)) == wave_idx
        ]
        if not wave_tiles:
            continue

        ppu_list = _ppu_summaries(wave_tiles)
        w_full_resident = all(
            w_fits_in_sram(
                p["local_m"],
                p["local_k"],
                p["local_n"],
                ppu_sram_kb=ppu_sram_kb,
            )
            for p in ppu_list
        )
        use_double_buffer = not w_full_resident
        k_chunk = min(
            compute_k_chunk_elems(
                p["local_m"],
                p["local_k"],
                p["local_n"],
                ppu_sram_kb=ppu_sram_kb,
                double_buffer=use_double_buffer,
            )
            for p in ppu_list
        )
        local_k = ppu_list[0]["local_k"]
        k_chunks = [local_k] if w_full_resident else split_k_chunks(local_k, k_chunk)

        a_bytes = m * k * FP16_BYTES
        heads_in_wave = int(wave.get("heads_in_wave", 1))
        if heads_in_wave > 1:
            a_bytes *= heads_in_wave
        a_load_cycles = cycles_for_bytes(a_bytes, lpddr_bpc)

        prefetch_cycles: list[int] = []
        compute_cycles: list[int] = []
        peak_lpddr_bpc = 0

        if w_full_resident:
            w_bytes = sum(p["local_n"] * local_k * FP16_BYTES for p in ppu_list)
            w_load_cycles = cycles_for_bytes(w_bytes, lpddr_bpc)
            prefetch_cycles = [w_load_cycles]
            compute_cycles = [_wave_compute_only_cycles(wave_tiles)]
            if lpddr_bpc > 0 and w_load_cycles > 0:
                peak_lpddr_bpc = math.ceil(w_bytes / w_load_cycles)
            pipeline = (
                a_load_cycles
                + w_load_cycles
                + compute_cycles[0]
            )
        else:
            for k_j in k_chunks:
                w_bytes = sum(p["local_n"] * k_j * FP16_BYTES for p in ppu_list)
                p_cy = cycles_for_bytes(w_bytes, lpddr_bpc)
                prefetch_cycles.append(p_cy)
                if lpddr_bpc > 0:
                    peak_lpddr_bpc = max(peak_lpddr_bpc, math.ceil(w_bytes / p_cy))

                c_cy = max(
                    compute_os_cycle_count(int(t["local_m"]), k_j, int(t["local_n"]))
                    for t in wave_tiles
                )
                compute_cycles.append(c_cy)

            pipeline = a_load_cycles + double_buffer_pipeline_cycles(
                prefetch_cycles, compute_cycles
            )
        c_bytes = sum(p["local_m"] * p["local_n"] * FP16_BYTES for p in ppu_list)
        c_writeback_cycles = cycles_for_bytes(c_bytes, lpddr_bpc)
        pipeline += c_writeback_cycles
        if lpddr_bpc > 0 and c_writeback_cycles > 0:
            peak_lpddr_bpc = max(
                peak_lpddr_bpc,
                math.ceil(c_bytes / c_writeback_cycles),
            )

        wave_compute_only = _wave_compute_only_cycles(wave_tiles)

        prefetch_only = sum(prefetch_cycles)
        compute_chunks = sum(compute_cycles)
        bottleneck = "lpddr" if pipeline > wave_compute_only else "compute"

        wave_stats.append(
            {
                "wave_index": wave_idx,
                "active_ppu_count": len(ppu_list),
                "k_chunk": k_chunk,
                "num_k_chunks": len(k_chunks),
                "a_bytes": a_bytes,
                "a_load_cycles": a_load_cycles,
                "prefetch_cycles_total": prefetch_only,
                "compute_chunk_cycles_total": compute_chunks,
                "c_bytes": c_bytes,
                "c_writeback_cycles": c_writeback_cycles,
                "pipeline_cycles": pipeline,
                "compute_only_cycles": wave_compute_only,
                "peak_lpddr_bytes_per_cycle": peak_lpddr_bpc,
                "bottleneck": bottleneck,
                "w_full_resident": w_full_resident,
                "ppus": [
                    {
                        **p,
                        "k_chunk": compute_k_chunk_elems(
                            p["local_m"],
                            p["local_k"],
                            p["local_n"],
                            ppu_sram_kb=ppu_sram_kb,
                            double_buffer=use_double_buffer,
                        ),
                        "a_bytes": p["local_m"] * p["local_k"] * FP16_BYTES,
                        "w_bytes": p["local_k"] * p["local_n"] * FP16_BYTES,
                        "w_buf_bytes": (
                            p["local_k"] * p["local_n"] * FP16_BYTES
                            if w_full_resident
                            else (
                                (ppu_sram_bytes(ppu_sram_kb) - p["local_m"] * p["local_k"] * FP16_BYTES)
                                // (2 if use_double_buffer else 1)
                            )
                        ),
                    }
                    for p in ppu_list
                ],
            }
        )
        total_pipeline += pipeline
        total_compute_only += wave_compute_only

    overall_bottleneck = "lpddr" if total_pipeline > compute_only_cycles else "compute"
    any_double_buffer = any(not w["w_full_resident"] for w in wave_stats)

    return {
        "ppu_sram_size_kb": ppu_sram_kb,
        "lpddr_bandwidth_gbps": lpddr_bandwidth_gbps,
        "clock_ghz": clock_ghz,
        "lpddr_bytes_per_cycle": lpddr_bpc,
        "double_buffer": any_double_buffer,
        "compute_only_cycles": compute_only_cycles,
        "lpddr_aware_cycles": total_pipeline,
        "bottleneck": overall_bottleneck,
        "waves": wave_stats,
    }

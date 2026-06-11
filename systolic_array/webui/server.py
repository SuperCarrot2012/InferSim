"""FastAPI server for systolic array Web UI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from systolic_array.simulator import aggregate_ppu_memory, render_micro_snapshot, run_simulation
from systolic_array.types import (
    DataflowType,
    SimResult,
    _macro_snapshot_to_dict,
    _micro_tile_to_dict,
)

app = FastAPI(title="InferSim Systolic Array Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sim_cache: dict[str, SimResult] = {}
_macro_cache: dict[str, list[dict[str, Any]]] = {}
_micro_cache: dict[str, dict[str, Any]] = {}


class SimulateRequest(BaseModel):
    m: int = Field(ge=1, le=64)
    k: int = Field(ge=1, le=64)
    n: int = Field(ge=1, le=12288)
    rows: int = Field(default=16, ge=1, le=64)
    cols: int = Field(default=16, ge=1, le=64)
    dataflow: str = Field(default="output_stationary")
    mac_latency: int = Field(default=1, ge=1)
    weight_load_cycles: int = Field(default=1, ge=0)


class SimulateResponse(BaseModel):
    sim_id: str
    total_cycles: int
    config: dict[str, Any]
    dims: dict[str, int]
    tile_plan: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    try:
        flow = DataflowType(req.dataflow)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown dataflow: {req.dataflow}") from exc

    try:
        result = run_simulation(
            req.m,
            req.k,
            req.n,
            rows=req.rows,
            cols=req.cols,
            dataflow=flow,
            mac_latency=req.mac_latency,
            weight_load_cycles=req.weight_load_cycles,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sim_id = str(uuid.uuid4())
    _sim_cache[sim_id] = result

    return SimulateResponse(
        sim_id=sim_id,
        total_cycles=result.total_cycles,
        config=result.config,
        dims=result.dims,
        tile_plan=result.tile_plan,
    )


def _macro_snapshots(sim_id: str) -> list[dict[str, Any]]:
    if sim_id not in _macro_cache:
        result = _sim_cache[sim_id]
        _macro_cache[sim_id] = [_macro_snapshot_to_dict(s) for s in result.snapshots]
    return _macro_cache[sim_id]


@app.get("/api/simulate/{sim_id}/snapshot/{cycle}")
def get_snapshot(sim_id: str, cycle: int) -> dict[str, Any]:
    if sim_id not in _sim_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    snapshots = _macro_snapshots(sim_id)
    if cycle < 0 or cycle >= len(snapshots):
        raise HTTPException(status_code=404, detail=f"Cycle {cycle} out of range")
    return snapshots[cycle]


@app.get("/api/simulate/{sim_id}/snapshots")
def get_snapshots(sim_id: str, from_cycle: int = 0, to_cycle: int | None = None) -> dict:
    if sim_id not in _sim_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    snapshots = _macro_snapshots(sim_id)
    end = to_cycle if to_cycle is not None else len(snapshots) - 1
    end = min(end, len(snapshots) - 1)
    if from_cycle < 0 or from_cycle >= len(snapshots):
        raise HTTPException(status_code=400, detail="Invalid from_cycle")
    return {"snapshots": snapshots[from_cycle : end + 1], "total": len(snapshots)}


@app.get("/api/simulate/{sim_id}/micro/{cycle}")
def get_micro_snapshot(sim_id: str, cycle: int, tile_key: str) -> dict[str, Any]:
    """On-demand Systolic Core View snapshot for one tile at a given cycle.

    tile_key: OS ``{ppu},{core_row},{core_col}``; WS ``{grid_row},{grid_col}``.
    """
    result = _sim_cache.get(sim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    try:
        snapshot = render_micro_snapshot(result, tile_key, cycle)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Tile {tile_key} not found") from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    effective = snapshot.cycle
    cache_key = f"{sim_id}:{tile_key}:{effective}"
    if cache_key not in _micro_cache:
        _micro_cache[cache_key] = _micro_tile_to_dict(snapshot)
    micro = _micro_cache[cache_key]

    return {
        "cycle": cycle,
        "effective_cycle": effective,
        "tile_key": tile_key,
        "micro": micro,
        "clamped": cycle != effective,
    }


@app.get("/api/simulate/{sim_id}/ppu_memory/{cycle}")
def get_ppu_memory(sim_id: str, cycle: int, ppu_index: int) -> dict[str, Any]:
    """Per-cycle memory sum across all systolic cores in one PPU."""
    result = _sim_cache.get(sim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    try:
        memory = aggregate_ppu_memory(result, ppu_index, cycle)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"cycle": cycle, "ppu_index": ppu_index, "memory": memory}


@app.get("/api/simulate/{sim_id}/meta")
def get_meta(sim_id: str) -> dict[str, Any]:
    result = _sim_cache.get(sim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {
        "config": result.config,
        "dims": result.dims,
        "total_cycles": result.total_cycles,
        "tile_plan": result.tile_plan,
    }


_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")

"""FastAPI server for systolic array Web UI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from systolic_array.simulator import run_simulation
from systolic_array.types import DataflowType

app = FastAPI(title="InferSim Systolic Array Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sim_cache: dict[str, dict[str, Any]] = {}


class SimulateRequest(BaseModel):
    m: int = Field(ge=1, le=64)
    k: int = Field(ge=1, le=64)
    n: int = Field(ge=1, le=64)
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
    _sim_cache[sim_id] = result.to_dict()

    return SimulateResponse(
        sim_id=sim_id,
        total_cycles=result.total_cycles,
        config=result.config,
        dims=result.dims,
    )


@app.get("/api/simulate/{sim_id}/snapshot/{cycle}")
def get_snapshot(sim_id: str, cycle: int) -> dict[str, Any]:
    data = _sim_cache.get(sim_id)
    if not data:
        raise HTTPException(status_code=404, detail="Simulation not found")
    snapshots = data["snapshots"]
    if cycle < 0 or cycle >= len(snapshots):
        raise HTTPException(status_code=404, detail=f"Cycle {cycle} out of range")
    return snapshots[cycle]


@app.get("/api/simulate/{sim_id}/snapshots")
def get_snapshots(sim_id: str, from_cycle: int = 0, to_cycle: int | None = None) -> dict:
    data = _sim_cache.get(sim_id)
    if not data:
        raise HTTPException(status_code=404, detail="Simulation not found")
    snapshots = data["snapshots"]
    end = to_cycle if to_cycle is not None else len(snapshots) - 1
    end = min(end, len(snapshots) - 1)
    if from_cycle < 0 or from_cycle >= len(snapshots):
        raise HTTPException(status_code=400, detail="Invalid from_cycle")
    return {"snapshots": snapshots[from_cycle : end + 1], "total": len(snapshots)}


@app.get("/api/simulate/{sim_id}/meta")
def get_meta(sim_id: str) -> dict[str, Any]:
    data = _sim_cache.get(sim_id)
    if not data:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {
        "config": data["config"],
        "dims": data["dims"],
        "total_cycles": data["total_cycles"],
    }


_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")

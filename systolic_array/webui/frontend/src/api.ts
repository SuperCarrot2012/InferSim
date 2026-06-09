import type { CycleSnapshot, DataflowType, SimulateResponse } from './types'

const API = '/api'

export async function runSimulation(body: {
  m: number
  k: number
  n: number
  rows: number
  cols: number
  dataflow: DataflowType
}): Promise<SimulateResponse> {
  const res = await fetch(`${API}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, mac_latency: 1, weight_load_cycles: 1 }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Simulation failed')
  }
  return res.json()
}

export async function fetchAllSnapshots(simId: string): Promise<CycleSnapshot[]> {
  const res = await fetch(`${API}/simulate/${simId}/snapshots?from_cycle=0`)
  if (!res.ok) throw new Error('Failed to load snapshots')
  const data = await res.json()
  return data.snapshots
}

export function coordLabel(prefix: string, coord: [number, number] | null): string {
  if (!coord) return '—'
  return `${prefix}[${coord[0]},${coord[1]}]`
}

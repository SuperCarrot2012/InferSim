export type DataflowType = 'output_stationary' | 'weight_stationary'

export interface PESnapshot {
  row: number
  col: number
  phase: string
  w_coord: [number, number] | null
  a_coord: [number, number] | null
  p_coord: [number, number] | null
  has_act: boolean
  has_weight?: boolean
  has_psum: boolean
  writeback?: boolean
  in_tile: boolean
}

export interface LinkAnim {
  from_row: number
  from_col: number
  to_row: number
  to_col: number
  label: string
  direction: 'right' | 'down'
}

export interface CycleSnapshot {
  cycle: number
  phase: string
  pes: PESnapshot[][]
  left_inject: (string | null)[]
  top_inject?: (string | null)[]
  bottom_output: (string | null)[]
  active_links: LinkAnim[]
  progress: Record<string, number>
  memory?: MemoryAccess
}

export interface MemoryAccess {
  dtype: string
  bytes_per_elem: number
  read_bytes: number
  write_bytes: number
  read_elems: number
  write_elems: number
  read_breakdown: { weight: number; activation: number }
  write_breakdown: { output: number }
}

export interface SimulateResponse {
  sim_id: string
  total_cycles: number
  config: Record<string, unknown>
  dims: { m: number; k: number; n: number }
}

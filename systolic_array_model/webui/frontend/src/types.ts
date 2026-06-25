export type DataflowType = 'output_stationary' | 'weight_stationary'

export interface DiePpuState {
  active: boolean
  computing: boolean
  done: boolean
  ppu_index: number
  ppu_row: number
  ppu_col: number
  active_cores: number
  label: string
}

export interface MacroArrayState {
  active: boolean
  computing: boolean
  done: boolean
  grid_row: number
  grid_col: number
  local_m: number
  local_k: number
  local_n: number
  m0: number
  k0: number
  n0: number
  label: string
}

export interface MicroTileSnapshot {
  phase: string
  pes: PESnapshot[][]
  left_inject: (string | null)[]
  top_inject?: (string | null)[]
  bottom_output: (string | null)[]
  active_links: LinkAnim[]
  progress: Record<string, number | string>
  memory?: MemoryAccess
}

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

/** Macro-only per-cycle snapshot (Logic Die / PPU views); PE grid fetched on demand. */
export interface CycleSnapshot {
  cycle: number
  phase: string
  pes: PESnapshot[][]
  left_inject: (string | null)[]
  top_inject?: (string | null)[]
  bottom_output: (string | null)[]
  active_links: LinkAnim[]
  progress: Record<string, number | string>
  memory?: MemoryAccess
  macro_arrays?: MacroArrayState[][]
  die_ppuss?: DiePpuState[][]
  ppu_core_grids?: Record<string, MacroArrayState[][]>
}

export interface TilePlan {
  hierarchy?: string
  die_rows?: number
  die_cols?: number
  ppu_count?: number
  ppu_core_grid?: number
  grid_rows?: number
  grid_cols?: number
  pe_rows: number
  pe_cols: number
  dataflow: string
  active_count: number
  active_ppu_count?: number
  tiles: Array<{
    ppu_index?: number
    ppu_row?: number
    ppu_col?: number
    core_row?: number
    core_col?: number
    grid_row?: number
    grid_col?: number
    m0: number
    k0: number
    n0: number
    local_m: number
    local_k: number
    local_n: number
    total_cycles?: number
  }>
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

export interface MemoryPeakStats {
  dtype: string
  bytes_per_elem: number
  peak_bytes: number
  peak_cycle: number
  read_bytes: number
  write_bytes: number
  read_breakdown: { weight: number; activation: number }
  write_breakdown: { output: number }
}

export interface SimulateResponse {
  sim_id: string
  total_cycles: number
  config: Record<string, unknown>
  dims: { m: number; k: number; n: number }
  tile_plan?: TilePlan
}

export interface GemmOpInfo {
  id: string
  label: string
  m: number
  k: number
  n: number
  note: string
  ppus_needed: number
  fits_die: boolean
}

export interface ModelInfo {
  id: string
  label: string
  description: string
  gemm_ops: GemmOpInfo[]
}

export interface ModelCatalog {
  models: ModelInfo[]
  default_model_id: string
  default_gemm_id: string
  hardware: {
    die_ppu_count: number
    os_m_max: number
    os_k_max: number
    os_n_max: number
    memory: string
  }
}

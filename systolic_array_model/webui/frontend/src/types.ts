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
  macs: MacSnapshot[][]
  left_inject: (string | null)[]
  top_inject?: (string | null)[]
  bottom_output: (string | null)[]
  active_links: LinkAnim[]
  progress: Record<string, number | string>
  memory?: MemoryAccess
}

export interface MacSnapshot {
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

/** Macro-only per-cycle snapshot (Logic Die / PPU views); MAC grid fetched on demand. */
export interface CycleSnapshot {
  cycle: number
  phase: string
  macs: MacSnapshot[][]
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
  mac_rows: number
  mac_cols: number
  dataflow: string
  active_count: number
  active_ppu_count?: number
  wave_count?: number
  os_n_wave_max?: number
  waves?: Array<{
    wave_index: number
    n0: number
    wave_n: number
    cycle_start: number
    cycle_end: number
    active_count: number
    active_ppu_count: number
  }>
  tiles: Array<{
    ppu_index?: number
    ppu_row?: number
    ppu_col?: number
    core_row?: number
    core_col?: number
    grid_row?: number
    grid_col?: number
    wave_index?: number
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
  logic_die_sram_demand?: LogicDieSramDemand | null
  logic_die_lpddr_schedule?: LogicDieLpddrSchedule | null
}

export interface GemmSimStat {
  totalCycles: number | null
  pending?: boolean
  error?: string
}

export interface SimCacheEntry {
  response: SimulateResponse
  snapshots: CycleSnapshot[]
}

export interface GemmOpInfo {
  id: string
  label: string
  m: number
  k: number
  n: number
  batch: number
  heads_per_wave?: number
  note: string
  ppus_needed: number
  wave_count: number
  fits_die: boolean
  ref_cycles?: number
  ref_label?: string
}

export interface ModelInfo {
  id: string
  label: string
  description: string
  gemm_ops: GemmOpInfo[]
}

export interface LogicDieSramWaveDemand {
  wave_index: number
  per_ppu_peak_bytes: number
  active_ppu_count: number
  logic_die_peak_bytes: number
}

export interface LogicDieSramDemand {
  per_ppu_peak_bytes: number
  logic_die_peak_bytes: number
  active_ppu_count: number
  die_ppu_count: number
  waves?: LogicDieSramWaveDemand[]
}

export interface LogicDieLpddrWaveSchedule {
  wave_index: number
  active_ppu_count: number
  k_chunk: number
  num_k_chunks: number
  a_bytes: number
  a_load_cycles: number
  prefetch_cycles_total: number
  compute_chunk_cycles_total: number
  c_bytes: number
  c_writeback_cycles: number
  pipeline_cycles: number
  compute_only_cycles: number
  peak_lpddr_bytes_per_cycle: number
  bottleneck: 'lpddr' | 'compute'
  w_full_resident?: boolean
  ppus?: Array<{
    ppu_index: number
    local_m: number
    local_k: number
    local_n: number
    k_chunk: number
    a_bytes: number
    w_bytes: number
    w_buf_bytes: number
  }>
}

export interface LogicDieLpddrSchedule {
  ppu_sram_size_kb: number
  lpddr_bandwidth_gbps: number
  clock_ghz: number
  lpddr_bytes_per_cycle: number
  double_buffer: boolean
  compute_only_cycles: number
  lpddr_aware_cycles: number
  bottleneck: 'lpddr' | 'compute'
  waves: LogicDieLpddrWaveSchedule[]
}

export interface BenchmarkInfo {
  id: string
  label: string
  model_id: string
  note?: string
  gemm_cycles: Record<string, number>
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
    os_n_wave_max: number
    memory: string
  }
  benchmarks?: BenchmarkInfo[]
}

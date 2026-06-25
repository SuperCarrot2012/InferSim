import type { MacroArrayState, TilePlan } from './types'

export const DIE_PPU_COUNT = 32
export const PPU_CORE_COUNT = 16
export const CORE_PE = 16
const PE_PER_CORE = CORE_PE * CORE_PE

export const TC_GRID = 4

export interface LogicDieUtilStats {
  utilization: number
  actualMacs: number
  capacityMacs: number
  viewLabel: string
}

/** Actual MACs (M×K×N) / full hierarchy MAC slots over all simulation cycles. */
export function computeHierarchyUtilization(
  dims: { m: number; k: number; n: number },
  totalCycles: number,
  dataflow: string,
): LogicDieUtilStats | null {
  if (totalCycles <= 0) return null

  const actualMacs = dims.m * dims.k * dims.n
  const isOS = dataflow === 'output_stationary'
  const capacityMacs = isOS
    ? DIE_PPU_COUNT * PPU_CORE_COUNT * PE_PER_CORE * totalCycles
    : TC_GRID * TC_GRID * PE_PER_CORE * totalCycles
  const utilization = capacityMacs > 0 ? actualMacs / capacityMacs : 0

  return {
    utilization,
    actualMacs,
    capacityMacs,
    viewLabel: isOS ? 'Logic Die View' : 'Tensor Core View',
  }
}

export interface PpuViewStats {
  ppuIndex: number
  activeCores: number
  computingCores: number
  doneCores: number
  selectedCore: {
    row: number
    col: number
    label: string
    local_m: number
    local_k: number
    local_n: number
    m0: number
    k0: number
    n0: number
    computing: boolean
    done: boolean
    total_cycles?: number
  } | null
}

export function summarizePpuView(
  ppuIndex: number,
  cores: MacroArrayState[][] | null | undefined,
  selected: { row: number; col: number } | null,
  tilePlan?: TilePlan,
): PpuViewStats | null {
  if (!cores) return null

  let activeCores = 0
  let computingCores = 0
  let doneCores = 0
  let selectedCore: PpuViewStats['selectedCore'] = null

  for (const row of cores) {
    for (const cell of row) {
      if (!cell.active) continue
      activeCores++
      if (cell.computing) computingCores++
      if (cell.done) doneCores++
    }
  }

  if (selected) {
    const cell = cores[selected.row]?.[selected.col]
    if (cell?.active) {
      const tileKey = coreTileKey(ppuIndex, selected)
      const tileMeta = findTileByKey(tilePlan, tileKey)
      selectedCore = {
        row: selected.row,
        col: selected.col,
        label: cell.label,
        local_m: cell.local_m,
        local_k: cell.local_k,
        local_n: cell.local_n,
        m0: cell.m0,
        k0: cell.k0,
        n0: cell.n0,
        computing: cell.computing,
        done: cell.done,
        total_cycles: tileMeta ? tileCycleCount(tileMeta) : undefined,
      }
    }
  }

  return {
    ppuIndex,
    activeCores,
    computingCores,
    doneCores,
    selectedCore,
  }
}

export const DIE_PPU_COLS = 8

export type TileEntry = TilePlan['tiles'][number]

export function tileKeyFromEntry(tile: TileEntry): string {
  if (tile.ppu_index != null && tile.core_row != null && tile.core_col != null) {
    return `${tile.ppu_index},${tile.core_row},${tile.core_col}`
  }
  return `${tile.grid_row},${tile.grid_col}`
}

export function tileCycleCount(tile: TileEntry): number {
  if (tile.total_cycles != null) return tile.total_cycles
  return Math.max(0, tile.local_m + tile.local_k + tile.local_n - 3) + 1
}

export function findTileByKey(tilePlan: TilePlan | undefined, tileKey: string): TileEntry | null {
  if (!tilePlan?.tiles) return null
  return tilePlan.tiles.find((t) => tileKeyFromEntry(t) === tileKey) ?? null
}

/** Map global timeline cycle to a valid micro frame for this tile (tail-aware). */
export function effectiveMicroCycle(globalCycle: number, tileCycles: number): number {
  if (tileCycles <= 0) return 0
  return Math.min(Math.max(0, globalCycle), tileCycles - 1)
}

export function coreTileKey(ppuIndex: number, core: { row: number; col: number }): string {
  return `${ppuIndex},${core.row},${core.col}`
}

export function ppuIndexFromDie(row: number, col: number): number {
  return row * DIE_PPU_COLS + col
}

export function ppuCoresGrid(
  snap: { ppu_core_grids?: Record<string, MacroArrayState[][]> } | null,
  ppuIndex: number | null
): MacroArrayState[][] | null {
  if (!snap?.ppu_core_grids || ppuIndex === null) return null
  return snap.ppu_core_grids[String(ppuIndex)] ?? null
}

export function firstActivePpu(
  die: { active?: boolean; ppu_index?: number }[][] | null | undefined
): { row: number; col: number; index: number } | null {
  if (!die) return null
  for (let r = 0; r < die.length; r++) {
    for (let c = 0; c < die[r].length; c++) {
      if (die[r][c]?.active) {
        return { row: r, col: c, index: die[r][c].ppu_index ?? ppuIndexFromDie(r, c) }
      }
    }
  }
  return null
}

export function firstActiveCore(
  cores: MacroArrayState[][] | null
): { row: number; col: number } | null {
  if (!cores) return null
  for (let r = 0; r < cores.length; r++) {
    for (let c = 0; c < cores[r].length; c++) {
      if (cores[r][c]?.active) return { row: r, col: c }
    }
  }
  return null
}

export function firstActiveArray(
  macro: MacroArrayState[][] | null | undefined
): { row: number; col: number } | null {
  if (!macro) return null
  for (let r = 0; r < macro.length; r++) {
    for (let c = 0; c < macro[r].length; c++) {
      if (macro[r][c]?.active) return { row: r, col: c }
    }
  }
  return null
}

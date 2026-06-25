import type { CycleSnapshot, MicroTileSnapshot } from './types'

export function idleMicroTile(
  rows: number,
  cols: number,
  dataflow: string,
): MicroTileSnapshot {
  return {
    phase: 'done',
    pes: Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => ({
        row: r,
        col: c,
        phase: 'idle',
        w_coord: null,
        a_coord: null,
        p_coord: null,
        has_act: false,
        has_weight: false,
        has_psum: false,
        writeback: false,
        in_tile: false,
      })),
    ),
    left_inject: Array(rows).fill(null),
    top_inject: Array(cols).fill(null),
    bottom_output: Array(cols).fill(null),
    active_links: [],
    progress: { dataflow, active_rows: 0, active_cols: 0 },
  }
}

export function emptySnapshot(rows: number, cols: number): CycleSnapshot {
  return {
    cycle: 0,
    phase: 'idle',
    pes: Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => ({
        row: r,
        col: c,
        phase: 'idle',
        w_coord: null,
        a_coord: null,
        p_coord: null,
        has_act: false,
        has_psum: false,
        in_tile: false,
      }))
    ),
    left_inject: Array(rows).fill(null),
    top_inject: Array(cols).fill(null),
    bottom_output: Array(cols).fill(null),
    active_links: [],
    progress: {},
  }
}

import type { CycleSnapshot } from './types'

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

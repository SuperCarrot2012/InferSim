import { motion } from 'framer-motion'
import type { MacroArrayState } from '../types'

export const MACRO_CELL = 72
export const MACRO_GAP = 10

interface Props {
  macro: MacroArrayState[][] | null | undefined
  selected: { row: number; col: number } | null
  onSelect: (row: number, col: number) => void
  title?: string
  cellSize?: number
  gap?: number
}

export function MacroGrid({
  macro,
  selected,
  onSelect,
  title = 'PPU View · 4×4 Core',
  cellSize = MACRO_CELL,
  gap = MACRO_GAP,
}: Props) {
  const grid = macro ?? emptyMacroGrid()
  const rows = grid.length
  const cols = grid[0]?.length ?? 4
  const width = cols * (cellSize + gap) - gap + 24
  const height = rows * (cellSize + gap) - gap + 48

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="macro-grid-svg" width={width} height={height}>
      <text x={width / 2} y={16} fill="#94a3b8" fontSize={13} fontFamily="Inter" textAnchor="middle">
        {title}
      </text>

      {grid.map((row, r) =>
        row.map((cell, c) => {
          const x = 12 + c * (cellSize + gap)
          const y = 28 + r * (cellSize + gap)
          const isSelected = selected?.row === r && selected?.col === c
          const stroke = isSelected
            ? '#f472b6'
            : cell.active
              ? cell.computing
                ? '#22d3ee'
                : cell.done
                  ? '#64748b'
                  : '#34d399'
              : '#2a3548'
          const fill = cell.active
            ? cell.computing
              ? '#1e293b'
              : cell.done
                ? '#111827'
                : '#172033'
            : '#0a0e17'

          return (
            <g
              key={`${r}-${c}`}
              onClick={() => cell.active && onSelect(r, c)}
              style={{ cursor: cell.active ? 'pointer' : 'default' }}
            >
              <motion.rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx={10}
                fill={fill}
                stroke={stroke}
                strokeWidth={isSelected ? 3.5 : cell.active ? 2.5 : 1.5}
                animate={
                  cell.active && cell.computing
                    ? { strokeOpacity: [0.55, 1, 0.55] }
                    : { strokeOpacity: 1 }
                }
                transition={{ repeat: cell.active && cell.computing ? Infinity : 0, duration: 1.4 }}
              />
              <text
                x={x + cellSize / 2}
                y={y + cellSize / 2 - 6}
                fill={cell.active ? '#e2e8f0' : '#475569'}
                fontSize={11}
                fontFamily="JetBrains Mono"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                [{r},{c}]
              </text>
              {cell.active && (
                <text
                  x={x + cellSize / 2}
                  y={y + cellSize / 2 + 12}
                  fill={cell.computing ? '#22d3ee' : cell.done ? '#64748b' : '#94a3b8'}
                  fontSize={9}
                  fontFamily="JetBrains Mono"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {cell.computing ? '计算中' : cell.done ? '完成' : '—'}
                </text>
              )}
              {cell.active && cell.label && (
                <title>{cell.label}</title>
              )}
            </g>
          )
        })
      )}
    </svg>
  )
}

function emptyMacroGrid(): MacroArrayState[][] {
  return Array.from({ length: 4 }, (_, r) =>
    Array.from({ length: 4 }, (_, c) => ({
      active: false,
      computing: false,
      done: false,
      grid_row: r,
      grid_col: c,
      local_m: 0,
      local_k: 0,
      local_n: 0,
      m0: 0,
      k0: 0,
      n0: 0,
      label: '',
    }))
  )
}

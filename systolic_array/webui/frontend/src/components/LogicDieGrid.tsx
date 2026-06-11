import { motion } from 'framer-motion'
import type { DiePpuState } from '../types'

export const DIE_PPU_ROWS = 6
export const DIE_PPU_COLS = 8
export const DIE_CELL = 50
export const DIE_GAP = 7

interface Props {
  diePpus: DiePpuState[][] | null | undefined
  selected: { row: number; col: number } | null
  onSelect: (row: number, col: number) => void
}

export function LogicDieGrid({ diePpus, selected, onSelect }: Props) {
  const grid = diePpus ?? emptyDieGrid()
  const width = DIE_PPU_COLS * (DIE_CELL + DIE_GAP) - DIE_GAP + 20
  const height = DIE_PPU_ROWS * (DIE_CELL + DIE_GAP) - DIE_GAP + 44

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="die-grid-svg" width={width} height={height}>
      <text x={width / 2} y={16} fill="#94a3b8" fontSize={14} fontFamily="Inter" textAnchor="middle">
        Logic Die View · 48 PPU
      </text>

      {grid.map((row, r) =>
        row.map((cell, c) => {
          const x = 10 + c * (DIE_CELL + DIE_GAP)
          const y = 28 + r * (DIE_CELL + DIE_GAP)
          const isSelected = selected?.row === r && selected?.col === c
          const stroke = isSelected
            ? '#f472b6'
            : cell.active
              ? cell.computing
                ? '#a78bfa'
                : '#64748b'
              : '#2a3548'

          return (
            <g
              key={`${r}-${c}`}
              onClick={() => cell.active && onSelect(r, c)}
              style={{ cursor: cell.active ? 'pointer' : 'default' }}
            >
              <motion.rect
                x={x}
                y={y}
                width={DIE_CELL}
                height={DIE_CELL}
                rx={8}
                fill={cell.active ? '#151c2c' : '#0a0e17'}
                stroke={stroke}
                strokeWidth={isSelected ? 2.5 : cell.active ? 2 : 1}
                animate={
                  cell.active && cell.computing
                    ? { strokeOpacity: [0.5, 1, 0.5] }
                    : { strokeOpacity: 1 }
                }
                transition={{ repeat: cell.active && cell.computing ? Infinity : 0, duration: 1.4 }}
              />
              <text
                x={x + DIE_CELL / 2}
                y={y + DIE_CELL / 2 - 6}
                fill={cell.active ? '#cbd5e1' : '#475569'}
                fontSize={12}
                fontFamily="JetBrains Mono"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                P{cell.ppu_index}
              </text>
              {cell.active && (
                <text
                  x={x + DIE_CELL / 2}
                  y={y + DIE_CELL / 2 + 12}
                  fill="#a78bfa"
                  fontSize={10}
                  fontFamily="JetBrains Mono"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {cell.active_cores}c
                </text>
              )}
              {cell.active && cell.label && <title>{cell.label}</title>}
            </g>
          )
        })
      )}
    </svg>
  )
}

function emptyDieGrid(): DiePpuState[][] {
  return Array.from({ length: DIE_PPU_ROWS }, (_, r) =>
    Array.from({ length: DIE_PPU_COLS }, (_, c) => ({
      active: false,
      computing: false,
      done: false,
      ppu_index: r * DIE_PPU_COLS + c,
      ppu_row: r,
      ppu_col: c,
      active_cores: 0,
      label: '',
    }))
  )
}

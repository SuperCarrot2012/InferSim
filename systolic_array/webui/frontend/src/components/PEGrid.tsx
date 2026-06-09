import { motion } from 'framer-motion'
import type { CycleSnapshot, PESnapshot } from '../types'
import { coordLabel } from '../api'

/** Fixed PE layout (px) — do not scale with viewport */
export const PE_CELL = 104
export const PE_GAP = 8
const LEFT_PAD = 8
const LEFT_LABEL_WIDTH = 120
const INJECT_LANE = 44
const MARGIN_X = LEFT_PAD + LEFT_LABEL_WIDTH + INJECT_LANE
const TOP_PAD = 8
const TOP_LABEL_HEIGHT = 28
const TOP_INJECT_LANE = 44
const MARGIN_Y = TOP_PAD + TOP_LABEL_HEIGHT + TOP_INJECT_LANE
const FONT_SIZE = 20
const LABEL_FS = 22
const OUTPUT_GAP = PE_GAP
const OUTPUT_BOTTOM_MARGIN = 12

function linkColor(label: string): string {
  if (label.startsWith('W')) return '#a78bfa'
  if (label.startsWith('A')) return '#38bdf8'
  return '#34d399'
}

function downMarkerId(label: string): string {
  if (label.startsWith('W')) return 'arrow-down-w'
  if (label.startsWith('A')) return 'arrow-down-a'
  return 'arrow-down-p'
}

interface Props {
  snapshot: CycleSnapshot
  gridRows: number
  gridCols: number
  selected: { row: number; col: number } | null
  onSelect: (row: number, col: number) => void
}

export function PEGrid({ snapshot, gridRows, gridCols, selected, onSelect }: Props) {
  const activeRows = Number(snapshot.progress?.active_rows ?? gridRows)
  const activeCols = Number(snapshot.progress?.active_cols ?? gridCols)
  const isWS = String(snapshot.progress?.dataflow ?? '') === 'weight_stationary'

  const gridContentH = gridRows * PE_CELL + (gridRows - 1) * PE_GAP
  const hasBottomOutput = snapshot.bottom_output.some(Boolean)
  const overlayBottomOutput = isWS && hasBottomOutput

  const peX = (c: number) => MARGIN_X + c * (PE_CELL + PE_GAP)
  const peY = (r: number) => MARGIN_Y + r * (PE_CELL + PE_GAP)

  const fullActiveTile = activeRows >= gridRows

  const bottomLabelY = overlayBottomOutput
    ? fullActiveTile
      ? peY(gridRows - 1) + PE_CELL + OUTPUT_GAP
      : peY(activeRows) + PE_CELL / 2
    : MARGIN_Y + gridContentH + OUTPUT_GAP

  const width = gridCols * (PE_CELL + PE_GAP) - PE_GAP + MARGIN_X + 16
  const needsBottomPadding = hasBottomOutput && (!overlayBottomOutput || fullActiveTile)
  const height = needsBottomPadding
    ? bottomLabelY + LABEL_FS + OUTPUT_BOTTOM_MARGIN
    : MARGIN_Y + gridContentH + 8
  const injectLabelX = LEFT_PAD + LEFT_LABEL_WIDTH
  const injectArrowX1 = injectLabelX + 8
  const topLabelY = TOP_PAD + TOP_LABEL_HEIGHT / 2
  const topInjectArrowY1 = TOP_PAD + TOP_LABEL_HEIGHT + 8
  const topInject = snapshot.top_inject ?? []

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="pe-grid-svg"
      width={width}
      height={height}
    >
      <defs>
        <marker id="arrow-right" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
          <path d="M0,0 L10,6 L0,12" fill="#38bdf8" />
        </marker>
        <marker id="arrow-down-w" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
          <path d="M0,0 L10,6 L0,12" fill="#a78bfa" />
        </marker>
        <marker id="arrow-down-a" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
          <path d="M0,0 L10,6 L0,12" fill="#38bdf8" />
        </marker>
        <marker id="arrow-down-p" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
          <path d="M0,0 L10,6 L0,12" fill="#34d399" />
        </marker>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {snapshot.active_links.map((link, i) => {
        const color = linkColor(link.label)
        const x2 = peX(link.to_col)
        const y2 = peY(link.to_row) + PE_CELL / 2

        if (link.direction === 'right') {
          const x1 = link.from_col < 0 ? injectArrowX1 : peX(link.from_col) + PE_CELL
          return (
            <motion.line
              key={`link-${i}`}
              x1={x1}
              y1={y2}
              x2={x2 - 4}
              y2={y2}
              stroke={color}
              strokeWidth={2.4}
              markerEnd="url(#arrow-right)"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.85 }}
              transition={{ duration: 0.2 }}
            />
          )
        }

        const xCenter = x2 + PE_CELL / 2
        const y1 =
          link.from_row < 0
            ? topInjectArrowY1
            : peY(link.from_row) + PE_CELL
        const yEnd = link.from_row < 0 ? peY(link.to_row) + 4 : y2 - 4

        return (
          <motion.line
            key={`link-${i}`}
            x1={xCenter}
            y1={y1}
            x2={xCenter}
            y2={yEnd}
            stroke={color}
            strokeWidth={2.4}
            markerEnd={`url(#${downMarkerId(link.label)})`}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.85 }}
            transition={{ duration: 0.2 }}
          />
        )
      })}

      {topInject.map((label, c) =>
        c < gridCols && label ? (
          <text
            key={`top-${c}`}
            x={peX(c) + PE_CELL / 2}
            y={topLabelY}
            fill="#a78bfa"
            fontSize={LABEL_FS}
            fontFamily="JetBrains Mono"
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {label}
          </text>
        ) : null
      )}

      {snapshot.left_inject.map((label, r) =>
        r < gridRows && label ? (
          <text
            key={`inject-${r}`}
            x={injectLabelX}
            y={peY(r) + PE_CELL / 2}
            fill="#38bdf8"
            fontSize={LABEL_FS}
            fontFamily="JetBrains Mono"
            textAnchor="end"
            dominantBaseline="middle"
          >
            {label}
          </text>
        ) : null
      )}

      {snapshot.pes.map((row, r) =>
        row.map((pe, c) => (
          <PECell
            key={`${r}-${c}`}
            pe={pe}
            x={peX(c)}
            y={peY(r)}
            selected={selected?.row === r && selected?.col === c}
            onClick={() => onSelect(r, c)}
          />
        ))
      )}

      {hasBottomOutput &&
        snapshot.bottom_output.map((label, c) =>
          c < activeCols && label ? (
            <text
              key={`out-${c}`}
              x={peX(c) + PE_CELL / 2}
              y={bottomLabelY}
              fill="#fbbf24"
              fontSize={LABEL_FS}
              fontFamily="JetBrains Mono"
              textAnchor="middle"
              dominantBaseline={overlayBottomOutput && !fullActiveTile ? 'middle' : 'hanging'}
              style={overlayBottomOutput ? { paintOrder: 'stroke fill' } : undefined}
              stroke={overlayBottomOutput ? '#0a0e17' : undefined}
              strokeWidth={overlayBottomOutput ? 4 : undefined}
            >
              {label}
            </text>
          ) : null
        )}
    </svg>
  )
}

function PECell({
  pe,
  x,
  y,
  selected,
  onClick,
}: {
  pe: PESnapshot
  x: number
  y: number
  selected: boolean
  onClick: () => void
}) {
  const idle = !pe.in_tile
  const phaseColor = idle
    ? '#2a3548'
    : pe.phase === 'load_weight' || pe.phase === 'load_psum'
      ? '#a78bfa'
      : pe.phase === 'compute'
        ? '#22d3ee'
        : '#64748b'

  const line1 = y + PE_CELL * 0.34
  const line2 = y + PE_CELL * 0.58
  const line3 = y + PE_CELL * 0.82
  const macActive = !idle && pe.has_act && (pe.has_weight || !!pe.w_coord)
  const borderColor = selected ? '#f472b6' : pe.writeback ? '#fbbf24' : phaseColor

  return (
    <g onClick={onClick} style={{ cursor: idle ? 'default' : 'pointer' }}>
      <motion.rect
        x={x}
        y={y}
        width={PE_CELL}
        height={PE_CELL}
        rx={12}
        fill={idle ? '#0a0e17' : '#1e293b'}
        stroke={borderColor}
        strokeWidth={selected || pe.writeback ? 4 : idle ? 1.5 : 3}
        filter={macActive ? 'url(#glow)' : undefined}
        animate={{ strokeOpacity: pe.writeback ? [0.7, 1, 0.7] : macActive ? [0.6, 1, 0.6] : 1 }}
        transition={{ repeat: pe.writeback || macActive ? Infinity : 0, duration: 1.2 }}
      />
      {!idle && pe.w_coord && (
        <text x={x + PE_CELL / 2} y={line1} fill="#a78bfa" fontSize={FONT_SIZE} fontFamily="JetBrains Mono" textAnchor="middle">
          {coordLabel('W', pe.w_coord)}
        </text>
      )}
      {!idle && pe.has_act && pe.a_coord && (
        <text x={x + PE_CELL / 2} y={line2} fill="#38bdf8" fontSize={FONT_SIZE} fontFamily="JetBrains Mono" textAnchor="middle">
          {coordLabel('A', pe.a_coord)}
        </text>
      )}
      {!idle && pe.has_psum && pe.p_coord && (
        <text x={x + PE_CELL / 2} y={line3} fill="#34d399" fontSize={FONT_SIZE} fontFamily="JetBrains Mono" textAnchor="middle">
          {coordLabel('P', pe.p_coord)}
        </text>
      )}
      {!idle && pe.writeback && (
        <text x={x + PE_CELL - 6} y={y + 18} fill="#fbbf24" fontSize={14} fontFamily="JetBrains Mono" textAnchor="end">
          ↓
        </text>
      )}
    </g>
  )
}

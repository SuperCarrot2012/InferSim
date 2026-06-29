import { useEffect, useState } from 'react'
import type { TilePlan } from '../types'

function DimInput({
  value,
  min,
  max,
  onChange,
}: {
  value: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  const [text, setText] = useState(String(value))

  useEffect(() => {
    setText(String(value))
  }, [value])

  const commit = (raw: string) => {
    const parsed = parseInt(raw, 10)
    if (Number.isNaN(parsed)) {
      setText(String(value))
      return
    }
    const clamped = Math.min(max, Math.max(min, parsed))
    setText(String(clamped))
    if (clamped !== value) onChange(clamped)
  }

  return (
    <input
      type="text"
      inputMode="numeric"
      autoComplete="off"
      value={text}
      onChange={(e) => {
        const digits = e.target.value.replace(/\D/g, '')
        setText(digits)
        if (digits === '') return
        const parsed = parseInt(digits, 10)
        if (!Number.isNaN(parsed)) onChange(Math.min(max, Math.max(min, parsed)))
      }}
      onBlur={() => commit(text)}
    />
  )
}

interface Props {
  m: number
  k: number
  n: number
  dataflow: 'output_stationary' | 'weight_stationary'
  arrayRows: number
  arrayCols: number
  osMMax: number
  osKMax: number
  osNMax: number
  wsMaxDim: number
  tilePlan?: TilePlan | null
  loading: boolean
  onMChange: (v: number) => void
  onKChange: (v: number) => void
  onNChange: (v: number) => void
  onDataflowChange: (v: 'output_stationary' | 'weight_stationary') => void
  onRun: () => void
}

const DATAFLOW_LABELS = {
  output_stationary: '部分和驻留 (OS)',
  weight_stationary: '权重驻留 (WS)',
} as const

function osTileHint(n: number, macCols: number): string {
  const nSlices = Math.ceil(n / macCols)
  const ppus = Math.ceil(nSlices / 16)
  const cores = nSlices
  if (cores === 1) return '1 Core · 1 PPU'
  if (ppus === 1) return `N 切 ${cores} Core · 1 PPU`
  return `N 切 ${cores} Core · ${ppus} PPU`
}

export function ConfigPanel({
  m, k, n, dataflow, arrayRows, arrayCols, osMMax, osKMax, osNMax, wsMaxDim, tilePlan, loading,
  onMChange, onKChange, onNChange, onDataflowChange, onRun,
}: Props) {
  const isOS = dataflow === 'output_stationary'

  return (
    <div className="config-panel">
      <h3>GEMM 配置</h3>

      <label className="config-field">
        数据流模式
        <select
          value={dataflow}
          onChange={(e) => onDataflowChange(e.target.value as 'output_stationary' | 'weight_stationary')}
        >
          <option value="output_stationary">{DATAFLOW_LABELS.output_stationary}</option>
          <option value="weight_stationary">{DATAFLOW_LABELS.weight_stationary}</option>
        </select>
      </label>

      <div className="config-grid">
        <label>
          M (A 行数)
          <DimInput value={m} min={1} max={isOS ? osMMax : wsMaxDim} onChange={onMChange} />
        </label>
        <label>
          K (内维)
          <DimInput value={k} min={1} max={isOS ? osKMax : wsMaxDim} onChange={onKChange} />
        </label>
        <label>
          N (B 列数)
          <DimInput value={n} min={1} max={isOS ? osNMax : arrayCols} onChange={onNChange} />
        </label>
      </div>

      <div className="config-actions">
        <button className="primary full" onClick={onRun} disabled={loading}>
          {loading ? '仿真中…' : '开始仿真'}
        </button>
      </div>

      <div className="config-note">
        {isOS ? (
          <>
            <strong>OS 结构：</strong>Logic Die View → PPU View → Systolic Core View<br />
            <strong>约束：</strong>M ∈ [1,{osMMax}]；K ∈ [1,{osKMax.toLocaleString()}]（流式）；N 优先切分，最大 {osNMax.toLocaleString()}<br />
            <strong>分块：</strong>{osTileHint(n, arrayCols)}
            {tilePlan && (
              <>
                <br />
                <strong>当前：</strong>{tilePlan.active_ppu_count ?? 0} PPU · {tilePlan.active_count} Core
              </>
            )}
          </>
        ) : (
          <>
            <strong>WS：</strong>4×4 阵列，每块 {arrayRows}×{arrayCols} MAC<br />
            <strong>约束：</strong>K ≤ {arrayRows}，N ≤ {arrayCols}
            {tilePlan && (
              <>
                <br />
                <strong>当前：</strong>{tilePlan.active_count} 阵列激活
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

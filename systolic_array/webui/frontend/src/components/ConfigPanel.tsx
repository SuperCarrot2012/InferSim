interface Props {
  m: number
  k: number
  n: number
  dataflow: 'output_stationary' | 'weight_stationary'
  arrayRows: number
  arrayCols: number
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

export function ConfigPanel({
  m, k, n, dataflow, arrayRows, arrayCols, loading,
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
          <input type="number" min={1} max={isOS ? arrayRows : 64} value={m} onChange={(e) => onMChange(+e.target.value)} />
        </label>
        <label>
          K (内维)
          <input type="number" min={1} max={isOS ? 64 : arrayRows} value={k} onChange={(e) => onKChange(+e.target.value)} />
        </label>
        <label>
          N (B 列数)
          <input type="number" min={1} max={arrayCols} value={n} onChange={(e) => onNChange(+e.target.value)} />
        </label>
      </div>

      <div className="config-actions">
        <button className="primary full" onClick={onRun} disabled={loading}>
          {loading ? '仿真中…' : '开始仿真'}
        </button>
      </div>

      <div className="config-note">
        <strong>约束：</strong>
        {isOS ? `M ≤ ${arrayRows}` : `K ≤ ${arrayRows}`}，N ≤ {arrayCols}
      </div>
    </div>
  )
}

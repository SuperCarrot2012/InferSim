import type { DataflowType, GemmOpInfo, ModelCatalog, ModelInfo } from '../types'

const DATAFLOW_LABELS = {
  output_stationary: '部分和驻留 (OS)',
  weight_stationary: '权重驻留 (WS)',
} as const

interface Props {
  catalog: ModelCatalog | null
  selectedModelId: string
  selectedGemmId: string | null
  dataflow: DataflowType
  loading: boolean
  onModelSelect: (modelId: string) => void
  onGemmSelect: (gemm: GemmOpInfo) => void
  onDataflowChange: (v: DataflowType) => void
}

function findModel(catalog: ModelCatalog | null, modelId: string): ModelInfo | null {
  return catalog?.models.find((m) => m.id === modelId) ?? null
}

function findGemm(model: ModelInfo | null, gemmId: string | null): GemmOpInfo | null {
  if (!model || !gemmId) return null
  return model.gemm_ops.find((g) => g.id === gemmId) ?? null
}

export function ModelPanel({
  catalog,
  selectedModelId,
  selectedGemmId,
  dataflow,
  loading,
  onModelSelect,
  onGemmSelect,
  onDataflowChange,
}: Props) {
  const model = findModel(catalog, selectedModelId)
  const selectedGemm = findGemm(model, selectedGemmId)
  const hw = catalog?.hardware
  const isOS = dataflow === 'output_stationary'

  return (
    <div className="config-panel model-panel">
      <h3>模型选择</h3>

      <div className="model-btn-group">
        {catalog?.models.map((m) => (
          <button
            key={m.id}
            type="button"
            className={`model-btn${m.id === selectedModelId ? ' active' : ''}`}
            onClick={() => onModelSelect(m.id)}
            disabled={loading}
          >
            {m.label}
          </button>
        ))}
      </div>

      {model && (
        <p className="config-desc">{model.description}</p>
      )}

      <label className="config-field">
        数据流模式
        <select
          value={dataflow}
          onChange={(e) => onDataflowChange(e.target.value as DataflowType)}
          disabled={loading}
        >
          <option value="output_stationary">{DATAFLOW_LABELS.output_stationary}</option>
          <option value="weight_stationary">{DATAFLOW_LABELS.weight_stationary}</option>
        </select>
      </label>

      {model && (
        <>
          <h3 className="section-subhead">GEMM 运算</h3>
          <div className="gemm-btn-group">
            {model.gemm_ops.map((op) => (
              <button
                key={op.id}
                type="button"
                className={`gemm-btn${op.id === selectedGemmId ? ' active' : ''}${!op.fits_die && isOS ? ' warn' : ''}`}
                onClick={() => onGemmSelect(op)}
                disabled={loading}
                title={`M=${op.m} K=${op.k} N=${op.n}${!op.fits_die ? ' · 超出单 Die 容量' : ''}`}
              >
                <span className="gemm-label">{op.label}</span>
                <span className="gemm-dims mono">
                  {op.m}×{op.k}×{op.n}
                </span>
              </button>
            ))}
          </div>
        </>
      )}

      {loading && (
        <div className="config-actions">
          <button className="primary full" disabled>
            仿真中…
          </button>
        </div>
      )}

      <div className="config-note">
        {isOS ? (
          <>
            <strong>OS 结构：</strong>Logic Die ({hw?.die_ppu_count ?? 32} PPU) → PPU → Core<br />
            <strong>存储：</strong>Load/Store 均由 SRAM 承担，不含 DDR 层级<br />
            <strong>约束：</strong>M ∈ [1,{hw?.os_m_max ?? 16}]；K 流式；N 最大 {hw?.os_n_max?.toLocaleString() ?? '8,192'}
            {selectedGemm && (
              <>
                <br />
                <strong>当前：</strong>{selectedGemm.label} · M×K×N = {selectedGemm.m}×{selectedGemm.k}×{selectedGemm.n}
                {!selectedGemm.fits_die && isOS && (
                  <> · <span className="warn-text">需 {selectedGemm.ppus_needed} PPU，超出 {hw?.die_ppu_count ?? 32} PPU 上限</span></>
                )}
              </>
            )}
          </>
        ) : (
          <>
            <strong>WS：</strong>4×4 阵列，每块 16×16 PE<br />
            <strong>存储：</strong>SRAM only<br />
            <strong>约束：</strong>K ≤ 64，N ≤ 16
            {selectedGemm && (
              <>
                <br />
                <strong>当前：</strong>{selectedGemm.label} · {selectedGemm.m}×{selectedGemm.k}×{selectedGemm.n}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

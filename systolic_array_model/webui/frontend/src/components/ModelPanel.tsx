import type { BenchmarkInfo, DataflowType, GemmOpInfo, GemmSimStat, ModelCatalog, ModelInfo } from '../types'

interface Props {
  catalog: ModelCatalog | null
  selectedModelId: string
  selectedGemmId: string | null
  dataflow: DataflowType
  initLoading: boolean
  simRunning: boolean
  gemmStats: Record<string, GemmSimStat>
  onModelSelect: (modelId: string) => void
  onGemmSelect: (gemmId: string) => void
}

function findModel(catalog: ModelCatalog | null, modelId: string): ModelInfo | null {
  return catalog?.models.find((m) => m.id === modelId) ?? null
}

function formatCycles(stat: GemmSimStat | undefined, simRunning: boolean): string {
  if (stat?.error) return '失败'
  if (stat?.pending || simRunning) return '…'
  if (stat?.totalCycles != null) return stat.totalCycles.toLocaleString()
  return '—'
}

function findBenchmark(catalog: ModelCatalog | null, modelId: string): BenchmarkInfo | null {
  return catalog?.benchmarks?.find((b) => b.model_id === modelId) ?? null
}

function refCyclesForOp(op: GemmOpInfo, benchmark: BenchmarkInfo | null): number | undefined {
  return op.ref_cycles ?? benchmark?.gemm_cycles[op.id]
}

function refLabelForOp(op: GemmOpInfo, benchmark: BenchmarkInfo | null): string | undefined {
  return op.ref_label ?? benchmark?.label
}

function formatPerformancePercent(simCycles: number, refCycles: number): string {
  if (refCycles <= 0 || simCycles <= 0) return ''
  const pct = (refCycles / simCycles) * 100
  if (pct >= 1000) return `${Math.round(pct)}%`
  if (pct >= 100) return `${pct.toFixed(0)}%`
  return `${pct.toFixed(1)}%`
}

export function ModelPanel({
  catalog,
  selectedModelId,
  selectedGemmId,
  dataflow,
  initLoading,
  simRunning,
  gemmStats,
  onModelSelect,
  onGemmSelect,
}: Props) {
  const model = findModel(catalog, selectedModelId)
  const hw = catalog?.hardware
  const benchmark = findBenchmark(catalog, selectedModelId)
  const isOS = dataflow === 'output_stationary'
  const panelLocked = initLoading || simRunning

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
            disabled={panelLocked}
          >
            {m.label}
          </button>
        ))}
      </div>

      {model && (
        <p className="config-desc">{model.description}</p>
      )}

      {model && (
        <>
          <h3 className="section-subhead">GEMM 运算</h3>
          {simRunning && (
            <p className="config-desc init-hint">仿真进行中…</p>
          )}
          <div className="gemm-list">
            {model.gemm_ops.map((op) => {
              const stat = gemmStats[op.id]
              const ready = stat?.totalCycles != null && !stat?.error
              const refCycles = refCyclesForOp(op, benchmark)
              const refLabel = refLabelForOp(op, benchmark)
              return (
                <GemmCard
                  key={op.id}
                  op={op}
                  isOS={isOS}
                  selected={op.id === selectedGemmId}
                  cyclesLabel={formatCycles(stat, !!stat?.pending)}
                  refCycles={refCycles}
                  refLabel={refLabel}
                  simCycles={stat?.totalCycles ?? null}
                  hasError={!!stat?.error}
                  disabled={!ready}
                  onSelect={() => onGemmSelect(op.id)}
                />
              )
            })}
          </div>
        </>
      )}

      <div className="config-note">
        {isOS ? (
          <>
            <strong>OS 结构：</strong>Logic Die ({hw?.die_ppu_count ?? 32} PPU) → PPU → Core<br />
            <strong>存储：</strong>Load/Store 均由 SRAM 承担，不含 DDR 层级<br />
            <strong>约束：</strong>M ∈ [1,{hw?.os_m_max ?? 16}]；K 流式；单 wave N ≤ {hw?.os_n_wave_max?.toLocaleString() ?? '8,192'}（超出自动分批）
          </>
        ) : (
          <>
            <strong>WS：</strong>4×4 阵列，每块 16×16 MAC<br />
            <strong>存储：</strong>SRAM only<br />
            <strong>约束：</strong>K ≤ 64，N ≤ 16
          </>
        )}
      </div>
    </div>
  )
}

function GemmCard({
  op,
  isOS,
  selected,
  cyclesLabel,
  refCycles,
  refLabel,
  simCycles,
  hasError,
  disabled,
  onSelect,
}: {
  op: GemmOpInfo
  isOS: boolean
  selected: boolean
  cyclesLabel: string
  refCycles?: number
  refLabel?: string
  simCycles: number | null
  hasError: boolean
  disabled: boolean
  onSelect: () => void
}) {
  const showRatio = simCycles != null && refCycles != null && refCycles > 0 && !hasError
  return (
    <div className={`gemm-card${selected ? ' selected' : ''}${hasError ? ' error' : ''}`}>
      <button
        type="button"
        className="gemm-card-head"
        onClick={onSelect}
        disabled={disabled}
      >
        {op.label}
      </button>
      <div className="gemm-card-body mono">
        <div className="gemm-card-dims">
          <span>B {(op.batch ?? 1).toLocaleString()}</span>
          <span>M {op.m.toLocaleString()}</span>
          <span>K {op.k.toLocaleString()}</span>
          <span>N {op.n.toLocaleString()}</span>
        </div>
        {isOS && op.wave_count > 1 && (
          <div className="gemm-card-waves">
            {op.wave_count} waves
            {(op.batch ?? 1) > 1 && (
              <> · {op.heads_per_wave ?? op.batch} head × 1 PPU</>
            )}
          </div>
        )}
        <div className="gemm-card-cycles">
          <span className="gemm-detail-key">总 Cycle</span>
          <span className={`gemm-detail-val${hasError ? '' : ' accent'}`}>
            {cyclesLabel}
            {showRatio && (
              <span className="gemm-benchmark-pct"> ({formatPerformancePercent(simCycles, refCycles)})</span>
            )}
          </span>
        </div>
        {refCycles != null && (
          <div className="gemm-card-benchmark">
            <span className="gemm-detail-key">{refLabel ?? '参考'}</span>
            <span className="gemm-detail-val">{refCycles.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  )
}

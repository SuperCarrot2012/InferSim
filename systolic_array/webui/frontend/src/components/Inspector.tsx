import type { PESnapshot } from '../types'
import { coordLabel } from '../api'
import type { MemoryAccess } from '../types'
import { formatBytesPerCycle } from '../memory'
import type { PpuViewStats } from '../microSnapshot'

interface Props {
  pe: PESnapshot | null
  phase: string
  dataflow: string
  memory: MemoryAccess
  ppuView?: PpuViewStats | null
  ppuMemory?: (MemoryAccess & { contributing_cores?: number }) | null
  coreViewContext?: string
}

export function Inspector({ pe, phase, dataflow, memory, ppuView, ppuMemory, coreViewContext }: Props) {
  const phaseLabel: Record<string, string> = {
    weight_load: '权重加载',
    compute: '矩阵计算',
    drain: '结果排出',
    done: '完成',
  }

  return (
    <div className="inspector">
      <h3>状态检查器</h3>

      {ppuView && (
        <div className="inspector-section ppu-view-panel">
          <h4>PPU View</h4>
          <div className="stat-row">
            <span className="label">PPU</span>
            <span className="value mono">#{ppuView.ppuIndex}</span>
          </div>
          <div className="stat-row">
            <span className="label">活跃 Core</span>
            <span className="value mono">{ppuView.activeCores} / 16</span>
          </div>
          <div className="stat-row">
            <span className="label">计算中</span>
            <span className="value mono accent">{ppuView.computingCores}</span>
          </div>
          <div className="stat-row">
            <span className="label">已完成</span>
            <span className="value mono muted">{ppuView.doneCores}</span>
          </div>
          {ppuView.selectedCore ? (
            <>
              <div className="ppu-view-divider" />
              <div className="stat-row">
                <span className="label">选中 Core</span>
                <span className="value mono">[{ppuView.selectedCore.row},{ppuView.selectedCore.col}]</span>
              </div>
              <div className="stat-row">
                <span className="label">分块</span>
                <span className="value mono ppu-view-label">{ppuView.selectedCore.label}</span>
              </div>
              <div className="stat-row">
                <span className="label">局部维度</span>
                <span className="value mono">
                  M×K×N = {ppuView.selectedCore.local_m}×{ppuView.selectedCore.local_k}×{ppuView.selectedCore.local_n}
                </span>
              </div>
              {ppuView.selectedCore.total_cycles != null && (
                <div className="stat-row">
                  <span className="label">Core 周期</span>
                  <span className="value mono">{ppuView.selectedCore.total_cycles}</span>
                </div>
              )}
              <div className="stat-row">
                <span className="label">状态</span>
                <span className={`badge ${ppuView.selectedCore.computing ? 'phase-compute' : ppuView.selectedCore.done ? 'ppu-view-done' : ''}`}>
                  {ppuView.selectedCore.computing ? '计算中' : ppuView.selectedCore.done ? '已完成' : '—'}
                </span>
              </div>
            </>
          ) : (
            <p className="hint ppu-view-hint">点击 PPU View 网格选择 Core</p>
          )}

          {ppuMemory && (
            <>
              <div className="ppu-view-divider" />
              <MemoryStatsPanel memory={ppuMemory} />
            </>
          )}
        </div>
      )}

      <div className="inspector-section core-view-panel">
        <h4>Systolic Core View</h4>
        {coreViewContext && (
          <p className="view-context mono">{coreViewContext}</p>
        )}

        <div className="stat-row">
          <span className="label">阶段</span>
          <span className={`badge phase-${phase}`}>{phaseLabel[phase] ?? phase}</span>
        </div>

        <MemoryStatsPanel memory={memory} />

        {pe ? (
          <div className="pe-detail">
            <h5>PE [{pe.row}, {pe.col}]</h5>
            <div className="register-grid">
              <Register label="Weight (W)" coord={pe.w_coord} prefix="W" color="#a78bfa" />
              <Register label="Activation (A)" coord={pe.a_coord} prefix="A" color="#38bdf8" />
              <Register label="Partial Sum (P)" coord={pe.p_coord} prefix="P" color="#34d399" highlight />
            </div>
          </div>
        ) : (
          <p className="hint core-view-hint">点击 Systolic Core View 网格选择 PE</p>
        )}
      </div>

      <div className="legend">
        <h4>图例 · {dataflow === 'output_stationary' ? 'OS' : 'WS'}</h4>
        <div className="legend-item"><span className="dot" style={{ background: '#a78bfa' }} /> W[k,n] {dataflow === 'output_stationary' ? '垂直流动 ↓' : '权重驻留'}</div>
        <div className="legend-item"><span className="dot" style={{ background: '#38bdf8' }} /> A[m,k] 水平流动 →</div>
        <div className="legend-item"><span className="dot" style={{ background: '#34d399' }} /> P[m,n] {dataflow === 'output_stationary' ? '部分和驻留' : '垂直累加 ↓'}</div>
        {dataflow === 'output_stationary' && (
          <div className="legend-item"><span className="dot" style={{ background: '#fbbf24' }} /> P 算完即写回 memory</div>
        )}
        {dataflow === 'weight_stationary' && (
          <div className="legend-item"><span className="dot" style={{ background: '#fbbf24' }} /> 底部输出坐标</div>
        )}
      </div>
    </div>
  )
}

function MemoryStatsPanel({
  memory,
  note,
}: {
  memory: MemoryAccess
  note?: string
}) {
  return (
    <div className="memory-subpanel">
      <h5>访存需求{note ? ` · ${note}` : ''}</h5>
      <div className="memory-dtype mono">{memory.dtype.toUpperCase()} · {memory.bytes_per_elem} B/elem</div>
      <div className="memory-stat">
        <div className="memory-stat-label">读取</div>
        <div className="memory-stat-value mono read">{formatBytesPerCycle(memory.read_bytes)}</div>
        <div className="memory-breakdown">
          {memory.read_breakdown.weight > 0 && (
            <span>W {formatBytesPerCycle(memory.read_breakdown.weight)}</span>
          )}
          {memory.read_breakdown.activation > 0 && (
            <span>A {formatBytesPerCycle(memory.read_breakdown.activation)}</span>
          )}
          {memory.read_bytes === 0 && <span>—</span>}
        </div>
      </div>
      <div className="memory-stat">
        <div className="memory-stat-label">写入</div>
        <div className="memory-stat-value mono write">{formatBytesPerCycle(memory.write_bytes)}</div>
        <div className="memory-breakdown">
          {memory.write_breakdown.output > 0 ? (
            <span>C {formatBytesPerCycle(memory.write_breakdown.output)}</span>
          ) : (
            <span>—</span>
          )}
        </div>
      </div>
    </div>
  )
}

function Register({
  label,
  coord,
  prefix,
  color,
  highlight,
}: {
  label: string
  coord: [number, number] | null
  prefix: string
  color: string
  highlight?: boolean
}) {
  return (
    <div className={`register ${highlight ? 'highlight' : ''}`}>
      <div className="register-label" style={{ color }}>{label}</div>
      <div className="register-value mono">{coordLabel(prefix, coord)}</div>
    </div>
  )
}

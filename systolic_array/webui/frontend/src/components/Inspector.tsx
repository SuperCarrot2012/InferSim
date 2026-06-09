import type { PESnapshot } from '../types'
import { coordLabel } from '../api'
import type { MemoryAccess } from '../types'
import { formatBytesPerCycle } from '../memory'

interface Props {
  pe: PESnapshot | null
  phase: string
  dataflow: string
  memory: MemoryAccess
}

export function Inspector({ pe, phase, dataflow, memory }: Props) {
  const phaseLabel: Record<string, string> = {
    weight_load: '权重加载',
    compute: '矩阵计算',
    drain: '结果排出',
    done: '完成',
  }

  return (
    <div className="inspector">
      <h3>状态检查器</h3>

      <div className="inspector-section">
        <div className="stat-row">
          <span className="label">阶段</span>
          <span className={`badge phase-${phase}`}>{phaseLabel[phase] ?? phase}</span>
        </div>
      </div>

      <div className="inspector-section memory-panel">
        <h4>访存需求</h4>
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

      {pe ? (
        <div className="inspector-section pe-detail">
          <h4>PE [{pe.row}, {pe.col}]</h4>
          <div className="register-grid">
            <Register label="Weight (W)" coord={pe.w_coord} prefix="W" color="#a78bfa" />
            <Register label="Activation (A)" coord={pe.a_coord} prefix="A" color="#38bdf8" />
            <Register label="Partial Sum (P)" coord={pe.p_coord} prefix="P" color="#34d399" highlight />
          </div>
        </div>
      ) : (
        <p className="hint">点击 PE 格子查看坐标详情</p>
      )}

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

import type { MacSnapshot } from '../types'
import { coordLabel } from '../api'
import type { MemoryAccess, MemoryPeakStats } from '../types'
import { formatBytesPerCycle, formatStorageBytes } from '../memory'
import type { LogicDieUtilStats, PpuViewStats, CoreSelectionStats } from '../microSnapshot'
import { formatTflops } from '../microSnapshot'
import type { LogicDieLpddrSchedule, LogicDieSramDemand } from '../types'
import { bytesPerCycleToTbytesPerSec, formatTbytesPerSec } from '../memory'

interface Props {
  mac: MacSnapshot | null
  phase: string
  dataflow: string
  memory: MemoryAccess
  memoryPeak?: MemoryPeakStats | null
  ppuView?: PpuViewStats | null
  ppuMemory?: (MemoryAccess & { contributing_cores?: number }) | null
  ppuMemoryPeak?: MemoryPeakStats | null
  logicDieUtil?: LogicDieUtilStats | null
  coreViewContext?: string
  coreSelection?: CoreSelectionStats | null
  logicDieSramDemand?: LogicDieSramDemand | null
  logicDieLpddrSchedule?: LogicDieLpddrSchedule | null
  currentWaveIndex?: number
  waveCount?: number
}

export function Inspector({
  mac,
  phase,
  dataflow,
  memory,
  memoryPeak,
  ppuView,
  ppuMemory,
  ppuMemoryPeak,
  logicDieUtil,
  coreViewContext,
  coreSelection,
  logicDieSramDemand,
  logicDieLpddrSchedule,
  currentWaveIndex = 0,
  waveCount = 1,
}: Props) {
  const phaseLabel: Record<string, string> = {
    weight_load: '权重加载',
    compute: '矩阵计算',
    drain: '结果排出',
    done: '完成',
  }

  const lpddrForWave =
    logicDieLpddrSchedule?.waves?.find((w) => w.wave_index === currentWaveIndex)
    ?? logicDieLpddrSchedule?.waves?.[0]

  const ppuLpddr = lpddrForWave?.ppus?.find((p) => p.ppu_index === ppuView?.ppuIndex)

  const activePpuCount = lpddrForWave?.active_ppu_count ?? 0
  const ppuLpddrBytesPerCycle =
    logicDieLpddrSchedule && activePpuCount > 0
      ? logicDieLpddrSchedule.lpddr_bytes_per_cycle / activePpuCount
      : null

  const sramForWave =
    logicDieSramDemand?.waves?.find((w) => w.wave_index === currentWaveIndex)
    ?? logicDieSramDemand?.waves?.[0]

  const sramDiePeakBytes =
    sramForWave?.logic_die_peak_bytes ?? logicDieSramDemand?.logic_die_peak_bytes ?? 0
  const sramPpuPeakBytes =
    sramForWave?.per_ppu_peak_bytes ?? logicDieSramDemand?.per_ppu_peak_bytes ?? 0
  const sramActivePpus =
    sramForWave?.active_ppu_count ?? logicDieSramDemand?.active_ppu_count ?? 0

  return (
    <div className="inspector">
      <h3>状态检查器</h3>

      {logicDieUtil && (
        <div className="inspector-section logic-die-view-panel">
          <h4>{logicDieUtil.viewLabel}</h4>
          <div className="stat-row">
            <span className="label">频率</span>
            <span className="value mono">{logicDieUtil.clockGhz} GHz</span>
          </div>
          {logicDieLpddrSchedule && (
            <>
              <div className="stat-row section-start">
                <span className="label">总 Cycle 数</span>
                <span className="value mono">
                  {logicDieLpddrSchedule.lpddr_aware_cycles.toLocaleString()}
                </span>
              </div>
              <p className="logic-die-sram-detail mono">
                其中计算 {logicDieLpddrSchedule.compute_only_cycles.toLocaleString()} cycle
              </p>
              <div className="stat-row">
                <span className="label">瓶颈</span>
                <span className={`value mono ${logicDieLpddrSchedule.bottleneck === 'lpddr' ? 'accent' : ''}`}>
                  {logicDieLpddrSchedule.bottleneck === 'lpddr' ? 'LPDDR' : 'Compute'}
                </span>
              </div>
              <div className="stat-row">
                <span className="label">LPDDR 峰值</span>
                <span className="value mono logic-die-sram">
                  <span>
                    {logicDieLpddrSchedule.lpddr_bytes_per_cycle.toLocaleString()} B/cycle
                  </span>
                  <span>
                    {formatTbytesPerSec(
                      bytesPerCycleToTbytesPerSec(
                        logicDieLpddrSchedule.lpddr_bytes_per_cycle,
                        logicDieUtil.clockGhz,
                      ),
                    )}
                  </span>
                </span>
              </div>
              {logicDieSramDemand && sramDiePeakBytes > 0 && (
                <>
                  <div className="stat-row">
                    <span className="label">SRAM 带宽需求</span>
                    <span className="value mono logic-die-sram">
                      <span>{sramDiePeakBytes.toLocaleString()} B/cycle</span>
                      <span>
                        {formatTbytesPerSec(
                          bytesPerCycleToTbytesPerSec(
                            sramDiePeakBytes,
                            logicDieUtil.clockGhz,
                          ),
                        )}
                      </span>
                    </span>
                  </div>
                  <p className="logic-die-sram-detail mono">
                    单 PPU 峰值 {sramPpuPeakBytes.toLocaleString()} B/cycle ×{' '}
                    {sramActivePpus} 活跃 PPU
                    {waveCount > 1 && <> · Wave {currentWaveIndex + 1}/{waveCount}</>}
                  </p>
                </>
              )}
            </>
          )}
          {!logicDieLpddrSchedule && logicDieSramDemand && sramDiePeakBytes > 0 && (
            <>
              <div className="stat-row section-start">
                <span className="label">SRAM 带宽需求</span>
                <span className="value mono logic-die-sram">
                  <span>{sramDiePeakBytes.toLocaleString()} B/cycle</span>
                  <span>
                    {formatTbytesPerSec(
                      bytesPerCycleToTbytesPerSec(
                        sramDiePeakBytes,
                        logicDieUtil.clockGhz,
                      ),
                    )}
                  </span>
                </span>
              </div>
              <p className="logic-die-sram-detail mono">
                单 PPU 峰值 {sramPpuPeakBytes.toLocaleString()} B/cycle ×{' '}
                {sramActivePpus} 活跃 PPU
                {waveCount > 1 && <> · Wave {currentWaveIndex + 1}/{waveCount}</>}
              </p>
            </>
          )}
          <div className="stat-row section-start">
            <span className="label">峰值算力</span>
            <span className="value mono logic-die-peak">
              <span>{logicDieUtil.peakMacsPerCycle.toLocaleString()} MAC/cycle</span>
              <span>{formatTflops(logicDieUtil.peakTflops)}</span>
            </span>
          </div>
          <div className="stat-row">
            <span className="label">利用率</span>
            <span className="value mono logic-die-util">
              {(logicDieUtil.utilization * 100).toFixed(2)}%
            </span>
          </div>
          <p className="logic-die-util-detail mono">
            {logicDieUtil.actualMacs.toLocaleString()} / {logicDieUtil.capacityMacs.toLocaleString()} MAC
          </p>
        </div>
      )}

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
          {ppuView.ppuLocal && (
            <>
              <div className="stat-row section-start">
                <span className="label">分块</span>
                <span className="value mono ppu-view-label">{ppuView.ppuLocal.label}</span>
              </div>
              <div className="stat-row">
                <span className="label">局部维度</span>
                <span className="value mono">
                  M×K×N = {ppuView.ppuLocal.local_m}×{ppuView.ppuLocal.local_k}×{ppuView.ppuLocal.local_n}
                </span>
              </div>
              <SramCapacityPanel capacity={ppuView.ppuLocal.sramCapacity} />
              {ppuLpddr && logicDieLpddrSchedule && lpddrForWave && ppuLpddrBytesPerCycle != null && (
                <LpddrPrefetchPanel
                  ppuLpddr={ppuLpddr}
                  diePeakBytesPerCycle={logicDieLpddrSchedule.lpddr_bytes_per_cycle}
                  activePpuCount={activePpuCount}
                  ppuBytesPerCycle={ppuLpddrBytesPerCycle}
                  numKChunks={lpddrForWave.num_k_chunks}
                  wFullResident={lpddrForWave.w_full_resident ?? false}
                  waveIndex={currentWaveIndex}
                  waveCount={waveCount}
                />
              )}
            </>
          )}

          {!ppuView.selectedCore && (
            <p className="hint ppu-view-hint">点击 PPU View 网格选择 Core</p>
          )}

          {ppuMemory && (
            <MemoryStatsPanel memory={ppuMemory} peak={ppuMemoryPeak} />
          )}
        </div>
      )}

      <div className="inspector-section core-view-panel">
        <h4>Core View</h4>
        {coreViewContext && (
          <p className="view-context mono">{coreViewContext}</p>
        )}

        {coreSelection ? (
          <>
            <div className="stat-row">
              <span className="label">选中 Core</span>
              <span className="value mono">[{coreSelection.row},{coreSelection.col}]</span>
            </div>
            <div className="stat-row">
              <span className="label">Core 分块</span>
              <span className="value mono ppu-view-label">{coreSelection.label}</span>
            </div>
            <div className="stat-row">
              <span className="label">Core 维度</span>
              <span className="value mono">
                M×K×N = {coreSelection.local_m}×{coreSelection.local_k}×{coreSelection.local_n}
              </span>
            </div>
            <div className="stat-row">
              <span className="label">状态</span>
              <span className={`badge ${coreSelection.computing ? 'phase-compute' : coreSelection.done ? 'ppu-view-done' : ''}`}>
                {coreSelection.computing ? '计算中' : coreSelection.done ? '已完成' : '—'}
              </span>
            </div>
          </>
        ) : ppuView ? (
          <p className="hint core-view-hint">在 PPU View 中选择 Core</p>
        ) : null}

        <div className="stat-row section-start">
          <span className="label">阶段</span>
          <span className={`badge phase-${phase}`}>{phaseLabel[phase] ?? phase}</span>
        </div>

        <MemoryStatsPanel memory={memory} peak={memoryPeak} />

        {mac ? (
          <div className="mac-detail">
            <h5>MAC单元 [{mac.row}, {mac.col}]</h5>
            <div className="register-grid">
              <Register label="Weight (W)" coord={mac.w_coord} prefix="W" color="#a78bfa" />
              <Register label="Activation (A)" coord={mac.a_coord} prefix="A" color="#38bdf8" />
              <Register label="Partial Sum (P)" coord={mac.p_coord} prefix="P" color="#34d399" highlight />
            </div>
          </div>
        ) : (
          <p className="hint core-view-hint">点击 Core View 网格选择 MAC单元</p>
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

function LpddrPrefetchPanel({
  ppuLpddr,
  diePeakBytesPerCycle,
  activePpuCount,
  ppuBytesPerCycle,
  numKChunks,
  wFullResident,
  waveIndex,
  waveCount,
}: {
  ppuLpddr: {
    k_chunk: number
    local_k: number
    w_buf_bytes: number
  }
  diePeakBytesPerCycle: number
  activePpuCount: number
  ppuBytesPerCycle: number
  numKChunks: number
  wFullResident: boolean
  waveIndex: number
  waveCount: number
}) {
  return (
    <div className="memory-subpanel sram-capacity-panel">
      <h5>LPDDR → SRAM Prefetch</h5>
      <div className="memory-dtype mono">
        {wFullResident ? 'W 全驻留' : 'Double-buffer'} · A 全驻留 PPU SRAM
        {waveCount > 1 && <> · Wave {waveIndex + 1}/{waveCount}</>}
      </div>
      <div className="memory-section-label">当前 PPU</div>
      <div className="memory-stat">
        <div className="memory-stat-label">LPDDR 带宽</div>
        <div className="memory-stat-value mono read">
          {ppuBytesPerCycle.toLocaleString()} B/cycle
        </div>
        <div className="memory-breakdown mono">
          <span>
            Die 峰值 {diePeakBytesPerCycle.toLocaleString()} B/cycle ÷ {activePpuCount} 活跃 PPU
          </span>
        </div>
      </div>
      <div className="memory-stat">
        <div className="memory-stat-label">K 分块</div>
        <div className="memory-stat-value mono">
          {ppuLpddr.k_chunk} × {numKChunks}
        </div>
        <div className="memory-breakdown mono">
          <span>K = {ppuLpddr.local_k.toLocaleString()}</span>
          <span>每 buf {formatStorageBytes(ppuLpddr.w_buf_bytes)} W</span>
        </div>
      </div>
    </div>
  )
}

function SramCapacityPanel({
  capacity,
}: {
  capacity: {
    dtype: string
    bytes_per_elem: number
    activation_elems: number
    weight_elems: number
    activation_bytes: number
    weight_bytes: number
    total_bytes: number
  }
}) {
  return (
    <div className="memory-subpanel sram-capacity-panel">
      <h5>SRAM 容量需求</h5>
      <div className="memory-dtype mono">
        {capacity.dtype.toUpperCase()} · {capacity.bytes_per_elem} B/elem
      </div>
      <div className="memory-section-label">当前 PPU 分块</div>
      <div className="memory-stat">
        <div className="memory-stat-label">A 激活</div>
        <div className="memory-stat-value mono read">{formatStorageBytes(capacity.activation_bytes)}</div>
        <div className="memory-breakdown mono">
          M×K = {capacity.activation_elems.toLocaleString()} elem
        </div>
      </div>
      <div className="memory-stat">
        <div className="memory-stat-label">W 权重</div>
        <div className="memory-stat-value mono read">{formatStorageBytes(capacity.weight_bytes)}</div>
        <div className="memory-breakdown mono">
          K×N = {capacity.weight_elems.toLocaleString()} elem
        </div>
      </div>
      <div className="memory-peak-divider" />
      <div className="memory-stat">
        <div className="memory-stat-label">合计</div>
        <div className="memory-stat-value mono peak-total">{formatStorageBytes(capacity.total_bytes)}</div>
      </div>
    </div>
  )
}

function MemoryStatsPanel({
  memory,
  peak,
  note,
}: {
  memory: MemoryAccess
  peak?: MemoryPeakStats | null
  note?: string
}) {
  return (
    <div className="memory-subpanel">
      <h5>SRAM 访存{note ? ` · ${note}` : ''}</h5>
      <div className="memory-dtype mono">{memory.dtype.toUpperCase()} · {memory.bytes_per_elem} B/elem</div>
      <div className="memory-section-label">当前 Cycle</div>
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

      {peak && (
        <>
          <div className="memory-peak-divider" />
          <div className="memory-section-label">峰值带宽</div>
          <div className="memory-stat">
            <div className="memory-stat-value mono peak-total">{formatBytesPerCycle(peak.peak_bytes)}</div>
            <div className="memory-peak-cycle mono">@ cycle {peak.peak_cycle}</div>
            <div className="memory-breakdown">
              {peak.read_breakdown.weight > 0 && (
                <span>W {formatBytesPerCycle(peak.read_breakdown.weight)}</span>
              )}
              {peak.read_breakdown.activation > 0 && (
                <span>A {formatBytesPerCycle(peak.read_breakdown.activation)}</span>
              )}
              {peak.write_breakdown.output > 0 && (
                <span>C {formatBytesPerCycle(peak.write_breakdown.output)}</span>
              )}
              {peak.peak_bytes === 0 && <span>—</span>}
            </div>
          </div>
        </>
      )}
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

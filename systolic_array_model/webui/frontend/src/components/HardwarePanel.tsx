import type { DataflowType } from '../types'
import type { HardwarePreset } from '../hardwareConfig'

const DATAFLOW_LABELS = {
  output_stationary: '部分和驻留 (OS)',
  weight_stationary: '权重驻留 (WS)',
} as const

interface Props {
  preset: HardwarePreset
  dataflow: DataflowType
  simRunning: boolean
  canStart: boolean
  onPresetChange: (patch: Partial<HardwarePreset>) => void
  onDataflowChange: (v: DataflowType) => void
  onStartSimulation: () => void
  onStopSimulation: () => void
}

function NumField({
  label,
  value,
  min,
  step,
  unit,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min?: number
  step?: number
  unit?: string
  disabled?: boolean
  onChange: (v: number) => void
}) {
  return (
    <label className="config-field">
      {label}
      <div className="hw-field-row">
        <input
          type="number"
          min={min}
          step={step ?? 1}
          value={value}
          disabled={disabled}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (Number.isFinite(n)) onChange(n)
          }}
        />
        {unit && <span className="hw-unit mono">{unit}</span>}
      </div>
    </label>
  )
}

export function HardwarePanel({
  preset,
  dataflow,
  simRunning,
  canStart,
  onPresetChange,
  onDataflowChange,
  onStartSimulation,
  onStopSimulation,
}: Props) {
  const panelLocked = simRunning

  return (
    <div className="config-panel hardware-panel">
      <h3>硬件参数</h3>

      <NumField
        label="频率"
        value={preset.clockGhz}
        min={0.1}
        step={0.1}
        unit="GHz"
        disabled={panelLocked}
        onChange={(clockGhz) => onPresetChange({ clockGhz })}
      />

      <div className="hw-section">
        <h4 className="section-subhead">PPU SRAM</h4>
        <NumField
          label="容量"
          value={preset.ppuSramSizeKb}
          min={1}
          unit="KiB"
          disabled={panelLocked}
          onChange={(ppuSramSizeKb) => onPresetChange({ ppuSramSizeKb })}
        />
      </div>

      <div className="hw-section">
        <h4 className="section-subhead">LPDDR</h4>
        <NumField
          label="带宽"
          value={preset.lpddrBandwidthGBps}
          min={1}
          step={1}
          unit="GB/s"
          disabled={panelLocked}
          onChange={(lpddrBandwidthGBps) => onPresetChange({ lpddrBandwidthGBps })}
        />
      </div>

      <div className="hw-dataflow-section">
        <label className="config-field">
          数据流模式
          <select
            value={dataflow}
            disabled={panelLocked}
            onChange={(e) => onDataflowChange(e.target.value as DataflowType)}
          >
            <option value="output_stationary">{DATAFLOW_LABELS.output_stationary}</option>
            <option value="weight_stationary">{DATAFLOW_LABELS.weight_stationary}</option>
          </select>
        </label>
      </div>

      <div className="config-actions hw-sim-actions">
        <button
          type="button"
          className="primary full"
          disabled={!canStart || simRunning}
          onClick={onStartSimulation}
        >
          开始仿真
        </button>
        <button
          type="button"
          className="full hw-stop-btn"
          disabled={!simRunning}
          onClick={onStopSimulation}
        >
          结束仿真
        </button>
      </div>
    </div>
  )
}

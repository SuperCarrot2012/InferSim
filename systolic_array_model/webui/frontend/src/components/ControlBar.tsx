interface Props {
  frameIndex: number
  displayCycle: number
  maxCycle: number
  totalCycles: number
  playing: boolean
  speed: number
  onFrameChange: (index: number) => void
  onPlay: () => void
  onPause: () => void
  onStepBack: () => void
  onStepForward: () => void
  onSpeedChange: (s: number) => void
}

export function ControlBar({
  frameIndex,
  displayCycle,
  maxCycle,
  totalCycles,
  playing,
  speed,
  onFrameChange,
  onPlay,
  onPause,
  onStepBack,
  onStepForward,
  onSpeedChange,
}: Props) {
  const maxFrame = Math.max(0, totalCycles - 1)

  return (
    <div className="control-bar">
      <div className="control-buttons">
        <button onClick={onStepBack} disabled={frameIndex <= 0} title="上一 Cycle">
          ⏮
        </button>
        {playing ? (
          <button onClick={onPause} className="primary" title="暂停">
            ⏸ 暂停
          </button>
        ) : (
          <button onClick={onPlay} className="primary" title="播放">
            ▶ 播放
          </button>
        )}
        <button onClick={onStepForward} disabled={frameIndex >= maxFrame} title="下一 Cycle">
          ⏭
        </button>
      </div>

      <div className="timeline">
        <span className="cycle-label mono">
          Cycle <strong>{displayCycle}</strong> / {maxCycle}
        </span>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={frameIndex}
          onChange={(e) => {
            onPause()
            onFrameChange(Number(e.target.value))
          }}
          className="scrubber"
        />
      </div>

      <div className="speed-control">
        <label>速度</label>
        <select value={speed} onChange={(e) => onSpeedChange(Number(e.target.value))}>
          <option value={2000}>0.5×</option>
          <option value={1000}>1×</option>
          <option value={400}>2.5×</option>
          <option value={150}>6×</option>
        </select>
      </div>
    </div>
  )
}

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAllSnapshots, runSimulation } from './api'
import type { CycleSnapshot, DataflowType, SimulateResponse } from './types'
import { ConfigPanel } from './components/ConfigPanel'
import { ControlBar } from './components/ControlBar'
import { Inspector } from './components/Inspector'
import { PEGrid } from './components/PEGrid'
import { deriveMemoryAccess } from './memory'
import { emptySnapshot } from './emptySnapshot'
import './App.css'

const ARRAY_ROWS = 16
const ARRAY_COLS = 16
const EMPTY_GRID = emptySnapshot(ARRAY_ROWS, ARRAY_COLS)

export default function App() {
  const [m, setM] = useState(4)
  const [k, setK] = useState(4)
  const [n, setN] = useState(4)
  const [dataflow, setDataflow] = useState<DataflowType>('output_stationary')
  const [response, setResponse] = useState<SimulateResponse | null>(null)
  const [snapshots, setSnapshots] = useState<CycleSnapshot[]>([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<{ row: number; col: number } | null>(null)
  const [started, setStarted] = useState(false)
  const playRef = useRef<number | null>(null)

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPlaying(false)
    setStarted(false)
    setSnapshots([])
    setResponse(null)
    setSelected(null)
    setFrameIndex(0)

    try {
      const res = await runSimulation({ m, k, n, rows: ARRAY_ROWS, cols: ARRAY_COLS, dataflow })
      const snaps = await fetchAllSnapshots(res.sim_id)
      setResponse(res)
      setSnapshots(snaps)
      setStarted(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [m, k, n, dataflow])

  useEffect(() => {
    if (!playing || snapshots.length === 0) return
    playRef.current = window.setInterval(() => {
      setFrameIndex((i) => {
        if (i >= snapshots.length - 1) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, speed)
    return () => {
      if (playRef.current) clearInterval(playRef.current)
    }
  }, [playing, speed, snapshots.length])

  const snap = snapshots[frameIndex] ?? null
  const maxCycle = snapshots.length > 0 ? snapshots[snapshots.length - 1].cycle : 0
  const totalCycles = response?.total_cycles ?? snapshots.length
  const simDataflow = (response?.config?.dataflow as string) ?? 'output_stationary'
  const selectedPE =
    snap && selected ? snap.pes[selected.row]?.[selected.col] ?? null : null
  const gridSnapshot = snap ?? EMPTY_GRID

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <span className="logo">◈</span>
          <div>
            <h1>InferSim 脉动阵列仿真器</h1>
          </div>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="layout">
        <aside className="sidebar">
          <ConfigPanel
            m={m} k={k} n={n} dataflow={dataflow}
            arrayRows={ARRAY_ROWS} arrayCols={ARRAY_COLS}
            loading={loading}
            onMChange={setM}
            onKChange={setK}
            onNChange={setN}
            onDataflowChange={setDataflow}
            onRun={handleRun}
          />
        </aside>

        <main className="main">
          {started && snapshots.length > 0 && (
            <ControlBar
              frameIndex={frameIndex}
              displayCycle={snap?.cycle ?? 0}
              maxCycle={maxCycle}
              totalCycles={totalCycles}
              playing={playing}
              speed={speed}
              onFrameChange={setFrameIndex}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onStepBack={() => setFrameIndex((i) => Math.max(0, i - 1))}
              onStepForward={() => setFrameIndex((i) => Math.min(snapshots.length - 1, i + 1))}
              onSpeedChange={setSpeed}
            />
          )}

          <div className="grid-area">
            <PEGrid
              snapshot={gridSnapshot}
              gridRows={ARRAY_ROWS}
              gridCols={ARRAY_COLS}
              selected={started ? selected : null}
              onSelect={(r, c) => setSelected({ row: r, col: c })}
            />
          </div>
        </main>

        <aside className="inspector-panel">
          {started && snap && (
            <Inspector
              pe={selectedPE}
              phase={snap.phase}
              dataflow={simDataflow}
              memory={deriveMemoryAccess(snap)}
            />
          )}
        </aside>
      </div>
    </div>
  )
}

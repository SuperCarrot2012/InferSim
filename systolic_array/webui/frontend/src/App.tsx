import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchAllSnapshots,
  fetchCoreMemoryPeak,
  fetchMicroSnapshot,
  fetchPpuCoreGrid,
  fetchPpuMemory,
  fetchPpuMemoryPeak,
  runSimulation,
} from './api'
import type {
  CycleSnapshot,
  DataflowType,
  MacroArrayState,
  MemoryAccess,
  MemoryPeakStats,
  MicroTileSnapshot,
  SimulateResponse,
} from './types'
import { ConfigPanel } from './components/ConfigPanel'
import { ControlBar } from './components/ControlBar'
import { Inspector } from './components/Inspector'
import { LogicDieGrid } from './components/LogicDieGrid'
import { MacroGrid } from './components/MacroGrid'
import { PEGrid } from './components/PEGrid'
import { deriveMemoryAccess } from './memory'
import { emptySnapshot } from './emptySnapshot'
import {
  computeHierarchyUtilization,
  coreTileKey,
  findTileByKey,
  firstActiveArray,
  firstActivePpu,
  ppuIndexFromDie,
  summarizePpuView,
  tileCycleCount,
} from './microSnapshot'
import './App.css'

const ARRAY_ROWS = 16
const ARRAY_COLS = 16
const OS_M_MAX = 16
const OS_K_MAX = 65536
const OS_N_MAX = 48 * 16 * 16
const WS_MAX_DIM = 64
const EMPTY_GRID = emptySnapshot(ARRAY_ROWS, ARRAY_COLS)

function microToCycleSnapshot(
  macro: CycleSnapshot,
  micro: MicroTileSnapshot,
): CycleSnapshot {
  return {
    cycle: macro.cycle,
    phase: micro.phase,
    pes: micro.pes,
    left_inject: micro.left_inject,
    top_inject: micro.top_inject,
    bottom_output: micro.bottom_output,
    active_links: micro.active_links,
    progress: micro.progress,
    memory: micro.memory,
    macro_arrays: macro.macro_arrays,
    die_ppuss: macro.die_ppuss,
    ppu_core_grids: macro.ppu_core_grids,
  }
}

export default function App() {
  const [m, setM] = useState(16)
  const [k, setK] = useState(16)
  const [n, setN] = useState(16)
  const [dataflow, setDataflow] = useState<DataflowType>('output_stationary')
  const [response, setResponse] = useState<SimulateResponse | null>(null)
  const [snapshots, setSnapshots] = useState<CycleSnapshot[]>([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [loading, setLoading] = useState(false)
  const [microLoading, setMicroLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<{ row: number; col: number } | null>(null)
  const [focusedPpu, setFocusedPpu] = useState<{ row: number; col: number; index: number } | null>(null)
  const [focusedCore, setFocusedCore] = useState<{ row: number; col: number } | null>(null)
  const [focusedArray, setFocusedArray] = useState<{ row: number; col: number } | null>(null)
  const [microSnap, setMicroSnap] = useState<CycleSnapshot | null>(null)
  const [microClamped, setMicroClamped] = useState(false)
  const [ppuMemory, setPpuMemory] = useState<(MemoryAccess & { contributing_cores: number }) | null>(null)
  const [coreMemoryPeak, setCoreMemoryPeak] = useState<MemoryPeakStats | null>(null)
  const [ppuMemoryPeak, setPpuMemoryPeak] = useState<MemoryPeakStats | null>(null)
  const [ppuCores, setPpuCores] = useState<MacroArrayState[][] | null>(null)
  const [started, setStarted] = useState(false)
  const playRef = useRef<number | null>(null)
  const microCache = useRef<Map<string, MicroTileSnapshot>>(new Map())
  const ppuGridCache = useRef<Map<string, MacroArrayState[][]>>(new Map())
  const ppuMemoryCache = useRef<Map<string, MemoryAccess & { contributing_cores: number }>>(new Map())
  const coreMemoryPeakCache = useRef<Map<string, MemoryPeakStats>>(new Map())
  const ppuMemoryPeakCache = useRef<Map<string, MemoryPeakStats>>(new Map())

  const isOS = dataflow === 'output_stationary'

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPlaying(false)
    setStarted(false)
    setSnapshots([])
    setResponse(null)
    setSelected(null)
    setFocusedPpu(null)
    setFocusedCore(null)
    setFocusedArray(null)
    setMicroSnap(null)
    setPpuMemory(null)
    setCoreMemoryPeak(null)
    setPpuMemoryPeak(null)
    setPpuCores(null)
    microCache.current.clear()
    ppuGridCache.current.clear()
    ppuMemoryCache.current.clear()
    coreMemoryPeakCache.current.clear()
    ppuMemoryPeakCache.current.clear()
    setFrameIndex(0)

    try {
      const res = await runSimulation({ m, k, n, rows: ARRAY_ROWS, cols: ARRAY_COLS, dataflow })
      const snaps = await fetchAllSnapshots(res.sim_id)
      setResponse(res)
      setSnapshots(snaps)

      if (dataflow === 'output_stationary') {
        const ppu = firstActivePpu(snaps[0]?.die_ppuss)
        setFocusedPpu(ppu)
        if (ppu && res.tile_plan?.tiles?.length) {
          const tile = res.tile_plan.tiles.find((t) => t.ppu_index === ppu.index) ?? res.tile_plan.tiles[0]
          if (tile.core_row != null && tile.core_col != null) {
            setFocusedCore({ row: tile.core_row, col: tile.core_col })
          }
        }
      } else {
        const arr = firstActiveArray(snaps[0]?.macro_arrays)
        setFocusedArray(arr)
      }
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

  useEffect(() => {
    if (!response?.sim_id || !snap || !started) {
      setMicroSnap(null)
      return
    }

    let tileKey: string | null = null
    if (isOS) {
      if (!focusedPpu || !focusedCore) {
        setMicroSnap(null)
        return
      }
      tileKey = coreTileKey(focusedPpu.index, focusedCore)
    } else {
      if (!focusedArray) {
        setMicroSnap(null)
        return
      }
      tileKey = `${focusedArray.row},${focusedArray.col}`
    }

    const tileMeta = findTileByKey(response.tile_plan, tileKey)
    const tileCycles = tileMeta
      ? tileCycleCount(tileMeta)
      : (response.total_cycles ?? snapshots.length)
    const clamped = snap.cycle >= tileCycles
    const cacheKey = `${response.sim_id}:${snap.cycle}:${tileKey}`
    const cached = microCache.current.get(cacheKey)
    if (cached) {
      setMicroSnap(microToCycleSnapshot(snap, cached))
      setMicroClamped(clamped)
      return
    }

    let cancelled = false
    setMicroLoading(true)
    fetchMicroSnapshot(response.sim_id, snap.cycle, tileKey)
      .then((micro) => {
        if (cancelled) return
        microCache.current.set(cacheKey, micro)
        setMicroSnap(microToCycleSnapshot(snap, micro))
        setMicroClamped(clamped)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Micro load failed')
      })
      .finally(() => {
        if (!cancelled) setMicroLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, response?.tile_plan, response?.total_cycles, snap, frameIndex, snapshots.length, isOS, focusedPpu, focusedCore, focusedArray, started])

  useEffect(() => {
    if (!response?.sim_id || !snap || !started || !isOS || !focusedPpu) {
      setPpuCores(null)
      return
    }

    const cacheKey = `${response.sim_id}:${focusedPpu.index}:${snap.cycle}`
    const cached = ppuGridCache.current.get(cacheKey)
    if (cached) {
      setPpuCores(cached)
      return
    }

    let cancelled = false
    fetchPpuCoreGrid(response.sim_id, snap.cycle, focusedPpu.index)
      .then((cores) => {
        if (cancelled) return
        ppuGridCache.current.set(cacheKey, cores)
        setPpuCores(cores)
      })
      .catch(() => {
        if (!cancelled) setPpuCores(null)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, snap, frameIndex, isOS, focusedPpu, started])

  useEffect(() => {
    if (!response?.sim_id || !snap || !started || !isOS || !focusedPpu) {
      setPpuMemory(null)
      return
    }

    const cacheKey = `${response.sim_id}:${focusedPpu.index}:${snap.cycle}`
    const cached = ppuMemoryCache.current.get(cacheKey)
    if (cached) {
      setPpuMemory(cached)
      return
    }

    let cancelled = false
    fetchPpuMemory(response.sim_id, snap.cycle, focusedPpu.index)
      .then((mem) => {
        if (cancelled) return
        ppuMemoryCache.current.set(cacheKey, mem)
        setPpuMemory(mem)
      })
      .catch(() => {
        if (!cancelled) setPpuMemory(null)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, snap, frameIndex, isOS, focusedPpu, started])

  useEffect(() => {
    if (!response?.sim_id || !started) {
      setCoreMemoryPeak(null)
      return
    }

    let tileKey: string | null = null
    if (isOS) {
      if (!focusedPpu || !focusedCore) {
        setCoreMemoryPeak(null)
        return
      }
      tileKey = coreTileKey(focusedPpu.index, focusedCore)
    } else {
      if (!focusedArray) {
        setCoreMemoryPeak(null)
        return
      }
      tileKey = `${focusedArray.row},${focusedArray.col}`
    }

    const cacheKey = `${response.sim_id}:${tileKey}`
    const cached = coreMemoryPeakCache.current.get(cacheKey)
    if (cached) {
      setCoreMemoryPeak(cached)
      return
    }

    let cancelled = false
    fetchCoreMemoryPeak(response.sim_id, tileKey)
      .then((peak) => {
        if (cancelled) return
        coreMemoryPeakCache.current.set(cacheKey, peak)
        setCoreMemoryPeak(peak)
      })
      .catch(() => {
        if (!cancelled) setCoreMemoryPeak(null)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, isOS, focusedPpu, focusedCore, focusedArray, started])

  useEffect(() => {
    if (!response?.sim_id || !started || !isOS || !focusedPpu) {
      setPpuMemoryPeak(null)
      return
    }

    const cacheKey = `${response.sim_id}:${focusedPpu.index}`
    const cached = ppuMemoryPeakCache.current.get(cacheKey)
    if (cached) {
      setPpuMemoryPeak(cached)
      return
    }

    let cancelled = false
    fetchPpuMemoryPeak(response.sim_id, focusedPpu.index)
      .then((peak) => {
        if (cancelled) return
        ppuMemoryPeakCache.current.set(cacheKey, peak)
        setPpuMemoryPeak(peak)
      })
      .catch(() => {
        if (!cancelled) setPpuMemoryPeak(null)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, isOS, focusedPpu, started])

  const maxCycle = snapshots.length > 0 ? snapshots[snapshots.length - 1].cycle : 0
  const totalCycles = response?.total_cycles ?? snapshots.length
  const simDataflow = (response?.config?.dataflow as string) ?? 'output_stationary'
  const tilePlan = response?.tile_plan

  const ppuCoresForView = isOS ? ppuCores : snap?.macro_arrays
  const ppuViewStats = isOS && focusedPpu
    ? summarizePpuView(focusedPpu.index, ppuCoresForView, focusedCore, tilePlan)
    : null

  const logicDieUtil = response?.dims
    ? computeHierarchyUtilization(response.dims, totalCycles, simDataflow)
    : null

  const selectedPE =
    microSnap && selected ? microSnap.pes[selected.row]?.[selected.col] ?? null : null
  const gridSnapshot = microSnap ?? EMPTY_GRID

  const coreViewContext = (() => {
    const doneSuffix = microClamped ? ' · 已完成' : ''
    if (isOS && focusedPpu && focusedCore) {
      return `PPU ${focusedPpu.index} · Core [${focusedCore.row},${focusedCore.col}]${doneSuffix}`
    }
    if (focusedArray) {
      return `Array [${focusedArray.row},${focusedArray.col}]${doneSuffix}`
    }
    return undefined
  })()

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
            osMMax={OS_M_MAX} osKMax={OS_K_MAX} osNMax={OS_N_MAX} wsMaxDim={WS_MAX_DIM}
            tilePlan={tilePlan}
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
            {isOS ? (
              <div className="viz-hierarchy">
                <div className="viz-macro-column">
                  <section className="viz-layer viz-die">
                    <div className="viz-panel-head">
                      <h4>Logic Die View</h4>
                      {tilePlan && (
                        <span className="viz-badge accent">
                          {tilePlan.active_ppu_count ?? 0} / 48 PPU
                        </span>
                      )}
                    </div>
                    <LogicDieGrid
                      diePpus={snap?.die_ppuss}
                      selected={focusedPpu ? { row: focusedPpu.row, col: focusedPpu.col } : null}
                      onSelect={(r, c) => {
                        const idx = ppuIndexFromDie(r, c)
                        setFocusedPpu({ row: r, col: c, index: idx })
                        const tile = tilePlan?.tiles?.find((t) => t.ppu_index === idx)
                        if (tile?.core_row != null && tile.core_col != null) {
                          setFocusedCore({ row: tile.core_row, col: tile.core_col })
                        } else {
                          setFocusedCore(null)
                        }
                        setSelected(null)
                      }}
                    />
                  </section>

                  <section className="viz-layer viz-ppu">
                    <div className="viz-panel-head">
                      <h4>PPU View</h4>
                      {focusedPpu && (
                        <span className="viz-badge">PPU {focusedPpu.index}</span>
                      )}
                    </div>
                    <MacroGrid
                      macro={ppuCoresForView}
                      selected={focusedCore}
                      title="PPU View · 4×4 Core"
                      onSelect={(r, c) => {
                        setFocusedCore({ row: r, col: c })
                        setSelected(null)
                      }}
                    />
                  </section>
                </div>

                <section className="viz-layer viz-core">
                  <div className="viz-panel-head">
                    <h4>Systolic Core View</h4>
                    {focusedCore && (
                      <span className="viz-badge">
                        Core [{focusedCore.row},{focusedCore.col}]
                        {microClamped ? ' · 已完成' : ''}
                        {microLoading ? ' · 加载中…' : ''}
                      </span>
                    )}
                  </div>
                  <PEGrid
                    snapshot={gridSnapshot}
                    gridRows={ARRAY_ROWS}
                    gridCols={ARRAY_COLS}
                    selected={started && microSnap ? selected : null}
                    onSelect={(r, c) => setSelected({ row: r, col: c })}
                  />
                </section>
              </div>
            ) : (
              <div className="viz-split">
                <section className="viz-micro">
                  <div className="viz-panel-head">
                    <h4>Systolic Core View</h4>
                    {focusedArray && (
                      <span className="viz-badge">
                        阵列 [{focusedArray.row},{focusedArray.col}]
                        {microClamped ? ' · 已完成' : ''}
                        {microLoading ? ' · 加载中…' : ''}
                      </span>
                    )}
                  </div>
                  <PEGrid
                    snapshot={gridSnapshot}
                    gridRows={ARRAY_ROWS}
                    gridCols={ARRAY_COLS}
                    selected={started && microSnap ? selected : null}
                    onSelect={(r, c) => setSelected({ row: r, col: c })}
                  />
                </section>
                <section className="viz-macro">
                  <div className="viz-panel-head">
                    <h4>Tensor Core View</h4>
                    {tilePlan && (
                      <span className="viz-badge accent">
                        {tilePlan.active_count} / 16 阵列激活
                      </span>
                    )}
                  </div>
                  <MacroGrid
                    macro={snap?.macro_arrays}
                    selected={focusedArray}
                    title="Tensor Core View · 4×4 Array"
                    onSelect={(r, c) => {
                      setFocusedArray({ row: r, col: c })
                      setSelected(null)
                    }}
                  />
                </section>
              </div>
            )}
          </div>
        </main>

        <aside className="inspector-panel">
          {started && microSnap && (
            <Inspector
              pe={selectedPE}
              phase={microSnap.phase}
              dataflow={simDataflow}
              memory={deriveMemoryAccess(microSnap)}
              memoryPeak={coreMemoryPeak}
              ppuView={ppuViewStats}
              ppuMemory={ppuMemory}
              ppuMemoryPeak={ppuMemoryPeak}
              logicDieUtil={logicDieUtil}
              coreViewContext={coreViewContext}
            />
          )}
        </aside>
      </div>
    </div>
  )
}

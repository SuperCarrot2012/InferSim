import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchAllSnapshots,
  fetchCoreMemoryPeak,
  fetchMicroSnapshot,
  fetchModels,
  fetchLogicDieSramDemand,
  fetchPpuCoreGrid,
  fetchPpuMemory,
  fetchPpuMemoryPeak,
  runModelSimulation,
} from './api'
import type {
  CycleSnapshot,
  DataflowType,
  LogicDieLpddrSchedule,
  LogicDieSramDemand,
  MacroArrayState,
  MemoryAccess,
  MemoryPeakStats,
  MicroTileSnapshot,
  ModelCatalog,
  SimCacheEntry,
  SimulateResponse,
  GemmSimStat,
} from './types'
import { ModelPanel } from './components/ModelPanel'
import { HardwarePanel } from './components/HardwarePanel'
import { ControlBar } from './components/ControlBar'
import { Inspector } from './components/Inspector'
import { LogicDieGrid } from './components/LogicDieGrid'
import { MacroGrid } from './components/MacroGrid'
import { MacGrid } from './components/MacGrid'
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
  waveIndexFromSnapshot,
  waveLocalCycleFromSnapshot,
} from './microSnapshot'
import './App.css'
import { DEFAULT_HARDWARE_PRESET, type HardwarePreset } from './hardwareConfig'

const ARRAY_ROWS = 16
const ARRAY_COLS = 16
const DIE_PPU_COUNT = 32
const EMPTY_GRID = emptySnapshot(ARRAY_ROWS, ARRAY_COLS)

function simCacheKey(
  modelId: string,
  gemmId: string,
  dataflow: DataflowType,
  hw: HardwarePreset,
): string {
  return `${modelId}:${gemmId}:${dataflow}:${hw.clockGhz}:${hw.ppuSramSizeKb}:${hw.lpddrBandwidthGBps}`
}

function hardwarePayload(hw: HardwarePreset) {
  return {
    clock_ghz: hw.clockGhz,
    ppu_sram_size_kb: hw.ppuSramSizeKb,
    lpddr_bandwidth_gbps: hw.lpddrBandwidthGBps,
  }
}

function microToCycleSnapshot(
  macro: CycleSnapshot,
  micro: MicroTileSnapshot,
): CycleSnapshot {
  return {
    cycle: macro.cycle,
    phase: micro.phase,
    macs: micro.macs,
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
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null)
  const [selectedModelId, setSelectedModelId] = useState('llama3-8b')
  const [selectedGemmId, setSelectedGemmId] = useState<string | null>('qo_proj')
  const [dataflow, setDataflow] = useState<DataflowType>('output_stationary')
  const [hardwarePreset, setHardwarePreset] = useState<HardwarePreset>(DEFAULT_HARDWARE_PRESET)
  const [response, setResponse] = useState<SimulateResponse | null>(null)
  const [snapshots, setSnapshots] = useState<CycleSnapshot[]>([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [initLoading, setInitLoading] = useState(true)
  const [simRunning, setSimRunning] = useState(false)
  const [gemmStats, setGemmStats] = useState<Record<string, GemmSimStat>>({})
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
  const [logicDieSramDemand, setLogicDieSramDemand] = useState<LogicDieSramDemand | null>(null)
  const [logicDieLpddrSchedule, setLogicDieLpddrSchedule] = useState<LogicDieLpddrSchedule | null>(null)
  const [started, setStarted] = useState(false)
  const playRef = useRef<number | null>(null)
  const simCacheRef = useRef<Map<string, SimCacheEntry>>(new Map())
  const simRunIdRef = useRef(0)
  const simRunningRef = useRef(false)
  const microCache = useRef<Map<string, MicroTileSnapshot>>(new Map())
  const ppuGridCache = useRef<Map<string, MacroArrayState[][]>>(new Map())
  const ppuMemoryCache = useRef<Map<string, MemoryAccess & { contributing_cores: number }>>(new Map())
  const coreMemoryPeakCache = useRef<Map<string, MemoryPeakStats>>(new Map())
  const ppuMemoryPeakCache = useRef<Map<string, MemoryPeakStats>>(new Map())

  const isOS = dataflow === 'output_stationary'
  const diePpuCount = catalog?.hardware.die_ppu_count ?? DIE_PPU_COUNT

  const clearViewCaches = useCallback(() => {
    microCache.current.clear()
    ppuGridCache.current.clear()
    ppuMemoryCache.current.clear()
    coreMemoryPeakCache.current.clear()
    ppuMemoryPeakCache.current.clear()
  }, [])

  const activateView = useCallback((entry: SimCacheEntry, flow: DataflowType, gemmId: string) => {
    clearViewCaches()
    setPlaying(false)
    setSelected(null)
    setMicroSnap(null)
    setPpuMemory(null)
    setCoreMemoryPeak(null)
    setPpuMemoryPeak(null)
    setPpuCores(null)
    setLogicDieSramDemand(
      flow === 'output_stationary' ? (entry.response.logic_die_sram_demand ?? null) : null,
    )
    setLogicDieLpddrSchedule(
      flow === 'output_stationary' ? (entry.response.logic_die_lpddr_schedule ?? null) : null,
    )
    setFrameIndex(0)
    setResponse(entry.response)
    setSnapshots(entry.snapshots)
    setSelectedGemmId(gemmId)

    if (flow === 'output_stationary') {
      const ppu = firstActivePpu(entry.snapshots[0]?.die_ppuss)
      setFocusedPpu(ppu)
      if (ppu && entry.response.tile_plan?.tiles?.length) {
        const tile = entry.response.tile_plan.tiles.find((t) => t.ppu_index === ppu.index)
          ?? entry.response.tile_plan.tiles[0]
        if (tile.core_row != null && tile.core_col != null) {
          setFocusedCore({ row: tile.core_row, col: tile.core_col })
        } else {
          setFocusedCore(null)
        }
      } else {
        setFocusedCore(null)
      }
      setFocusedArray(null)
    } else {
      const arr = firstActiveArray(entry.snapshots[0]?.macro_arrays)
      setFocusedArray(arr)
      setFocusedPpu(null)
      setFocusedCore(null)
    }
    setStarted(true)
  }, [clearViewCaches])

  const ensureEntrySnapshots = useCallback(async (entry: SimCacheEntry): Promise<CycleSnapshot[]> => {
    if (entry.snapshots.length > 0) return entry.snapshots
    const snaps = await fetchAllSnapshots(entry.response.sim_id)
    entry.snapshots = snaps
    return snaps
  }, [])

  const handleGemmSelect = useCallback(async (gemmId: string) => {
    const key = simCacheKey(selectedModelId, gemmId, dataflow, hardwarePreset)
    const entry = simCacheRef.current.get(key)
    if (!entry) return
    try {
      const snaps = await ensureEntrySnapshots(entry)
      activateView({ response: entry.response, snapshots: snaps }, dataflow, gemmId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load snapshots')
    }
  }, [selectedModelId, dataflow, hardwarePreset, activateView, ensureEntrySnapshots])

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setCatalog(data)
        setSelectedModelId(data.default_model_id)
        setSelectedGemmId(data.default_gemm_id)
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load models'))
      .finally(() => setInitLoading(false))
  }, [])

  useEffect(() => {
    if (!catalog) return

    simRunIdRef.current += 1
    simRunningRef.current = false
    setSimRunning(false)

    const model = catalog.models.find((m) => m.id === selectedModelId) ?? catalog.models[0]
    if (!model) return

    setStarted(false)
    setPlaying(false)
    setResponse(null)
    setSnapshots([])
    setGemmStats(
      Object.fromEntries(
        model.gemm_ops.map((op) => [op.id, { totalCycles: null } satisfies GemmSimStat]),
      ),
    )
  }, [catalog, selectedModelId, dataflow, hardwarePreset])

  const stopSimulation = useCallback(() => {
    if (!simRunningRef.current) return
    simRunIdRef.current += 1
    simRunningRef.current = false
    setSimRunning(false)
    setGemmStats((prev) => {
      const next: Record<string, GemmSimStat> = { ...prev }
      for (const [id, stat] of Object.entries(next)) {
        if (stat.pending) {
          next[id] = { totalCycles: null }
        }
      }
      return next
    })
  }, [])

  const startSimulation = useCallback(async () => {
    if (!catalog || simRunningRef.current) return

    const model = catalog.models.find((m) => m.id === selectedModelId) ?? catalog.models[0]
    if (!model) return

    const runId = ++simRunIdRef.current
    simRunningRef.current = true
    setSimRunning(true)
    setStarted(false)
    setError(null)
    setPlaying(false)

    const pendingStats = Object.fromEntries(
      model.gemm_ops.map((op) => [op.id, { totalCycles: null, pending: true } satisfies GemmSimStat]),
    )
    setGemmStats(pendingStats)

    try {
      for (const op of model.gemm_ops) {
        if (runId !== simRunIdRef.current) return
        const key = simCacheKey(model.id, op.id, dataflow, hardwarePreset)
        try {
          const res = await runModelSimulation({
            model_id: model.id,
            gemm_id: op.id,
            rows: ARRAY_ROWS,
            cols: ARRAY_COLS,
            dataflow,
            hardware: hardwarePayload(hardwarePreset),
          })
          if (runId !== simRunIdRef.current) return

          const entry: SimCacheEntry = { response: res, snapshots: [] }
          simCacheRef.current.set(key, entry)
          const cycles = res.logic_die_lpddr_schedule?.lpddr_aware_cycles ?? res.total_cycles
          setGemmStats((prev) => ({
            ...prev,
            [op.id]: { totalCycles: cycles },
          }))
        } catch (e) {
          if (runId !== simRunIdRef.current) return
          const message = e instanceof Error ? e.message : 'Simulation failed'
          setGemmStats((prev) => ({
            ...prev,
            [op.id]: { totalCycles: null, error: message },
          }))
        }
      }
    } finally {
      if (runId === simRunIdRef.current) {
        simRunningRef.current = false
        setSimRunning(false)
      }
    }
  }, [catalog, selectedModelId, dataflow, hardwarePreset])

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
      const waveIdx = waveIndexFromSnapshot(snap)
      tileKey = coreTileKey(waveIdx, focusedPpu.index, focusedCore)
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
    const waveLocal = waveLocalCycleFromSnapshot(snap)
    const clamped = waveLocal >= tileCycles
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
      tileKey = coreTileKey(waveIndexFromSnapshot(snap), focusedPpu.index, focusedCore)
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

  useEffect(() => {
    if (!response?.sim_id || !started || !isOS) {
      setLogicDieSramDemand(null)
      return
    }

    if (response.logic_die_sram_demand) {
      setLogicDieSramDemand(response.logic_die_sram_demand)
      return
    }

    let cancelled = false
    fetchLogicDieSramDemand(response.sim_id)
      .then((demand) => {
        if (!cancelled) setLogicDieSramDemand(demand)
      })
      .catch(() => {
        if (!cancelled) setLogicDieSramDemand(null)
      })

    return () => {
      cancelled = true
    }
  }, [response?.sim_id, response?.logic_die_sram_demand, isOS, started])

  const maxCycle = snapshots.length > 0 ? snapshots[snapshots.length - 1].cycle : 0
  const totalCycles = response?.total_cycles ?? snapshots.length
  const simDataflow = (response?.config?.dataflow as string) ?? 'output_stationary'
  const tilePlan = response?.tile_plan

  const ppuCoresForView = isOS ? ppuCores : snap?.macro_arrays
  const ppuViewStats = isOS && focusedPpu
    ? summarizePpuView(focusedPpu.index, ppuCoresForView, focusedCore, tilePlan, waveIndexFromSnapshot(snap))
    : null

  const logicDieUtil = response?.dims
    ? computeHierarchyUtilization(
        response.dims,
        logicDieLpddrSchedule?.lpddr_aware_cycles
          ?? response.logic_die_lpddr_schedule?.lpddr_aware_cycles
          ?? totalCycles,
        simDataflow,
        hardwarePreset.clockGhz,
      )
    : null

  const patchHardwarePreset = useCallback((patch: Partial<HardwarePreset>) => {
    setHardwarePreset((prev) => ({ ...prev, ...patch }))
  }, [])

  const selectedMac =
    microSnap && selected ? microSnap.macs[selected.row]?.[selected.col] ?? null : null
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
            <h1>模型仿真器</h1>
          </div>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="layout">
        <aside className="sidebar sidebar-split">
          <div className="sidebar-col">
            <ModelPanel
              catalog={catalog}
              selectedModelId={selectedModelId}
              selectedGemmId={selectedGemmId}
              dataflow={dataflow}
              initLoading={initLoading}
              simRunning={simRunning}
              gemmStats={gemmStats}
              onModelSelect={setSelectedModelId}
              onGemmSelect={handleGemmSelect}
            />
          </div>
          <div className="sidebar-col">
            <HardwarePanel
              preset={hardwarePreset}
              dataflow={dataflow}
              simRunning={simRunning}
              canStart={!!catalog && !initLoading}
              onPresetChange={patchHardwarePreset}
              onDataflowChange={setDataflow}
              onStartSimulation={() => { void startSimulation() }}
              onStopSimulation={stopSimulation}
            />
          </div>
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
                          {tilePlan.active_ppu_count ?? 0} / {diePpuCount} PPU
                          {(tilePlan.wave_count ?? 1) > 1 && (
                            <> · Wave {(snap?.progress?.wave_index as number ?? 0) + 1}/{tilePlan.wave_count}</>
                          )}
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
                    <h4>Core View</h4>
                    {focusedCore && (
                      <span className="viz-badge">
                        Core [{focusedCore.row},{focusedCore.col}]
                        {microClamped ? ' · 已完成' : ''}
                        {microLoading ? ' · 加载中…' : ''}
                      </span>
                    )}
                  </div>
                  <MacGrid
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
                    <h4>Core View</h4>
                    {focusedArray && (
                      <span className="viz-badge">
                        阵列 [{focusedArray.row},{focusedArray.col}]
                        {microClamped ? ' · 已完成' : ''}
                        {microLoading ? ' · 加载中…' : ''}
                      </span>
                    )}
                  </div>
                  <MacGrid
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
              mac={selectedMac}
              phase={microSnap.phase}
              dataflow={simDataflow}
              memory={deriveMemoryAccess(microSnap)}
              memoryPeak={coreMemoryPeak}
              ppuView={ppuViewStats}
              ppuMemory={ppuMemory}
              ppuMemoryPeak={ppuMemoryPeak}
              logicDieUtil={logicDieUtil}
              coreViewContext={coreViewContext}
              coreSelection={ppuViewStats?.selectedCore ?? null}
              logicDieSramDemand={logicDieSramDemand}
              logicDieLpddrSchedule={logicDieLpddrSchedule}
              currentWaveIndex={waveIndexFromSnapshot(snap)}
              waveCount={tilePlan?.wave_count ?? 1}
            />
          )}
        </aside>
      </div>
    </div>
  )
}

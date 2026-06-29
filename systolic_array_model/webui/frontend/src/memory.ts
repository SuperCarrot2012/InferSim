import type { CycleSnapshot, MemoryAccess } from './types'

const FP16_BYTES = 2

export function formatStorageBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KiB`
  return `${bytes} B`
}

export function formatBytesPerCycle(bytes: number): string {
  return `${bytes} B/cycle`
}

/** bytes/cycle × GHz → TB/s */
export function bytesPerCycleToTbytesPerSec(
  bytesPerCycle: number,
  clockGhz: number,
): number {
  return (bytesPerCycle * clockGhz * 1e9) / 1e12
}

export function formatTbytesPerSec(tbytesPerSec: number): string {
  if (!Number.isFinite(tbytesPerSec) || tbytesPerSec <= 0) return '0 TB/s'
  if (tbytesPerSec >= 100) return `${Math.round(tbytesPerSec)} TB/s`
  if (tbytesPerSec >= 10) return `${tbytesPerSec.toFixed(1)} TB/s`
  return `${tbytesPerSec.toFixed(2)} TB/s`
}

export function deriveMemoryAccess(snapshot: CycleSnapshot): MemoryAccess {
  if (snapshot.memory) {
    return snapshot.memory
  }

  const k = Number(snapshot.progress.k ?? snapshot.macs.length)
  const n = Number(snapshot.progress.n ?? snapshot.macs[0]?.length ?? 0)
  const readActivation = snapshot.left_inject.filter(Boolean).length
  const writeOutput = snapshot.bottom_output.filter(Boolean).length

  let readWeight = 0
  if (snapshot.phase === 'weight_load') {
    readWeight = k * n
  }

  const readElems = readWeight + readActivation
  const writeElems = writeOutput

  return {
    dtype: 'fp16',
    bytes_per_elem: FP16_BYTES,
    read_bytes: readElems * FP16_BYTES,
    write_bytes: writeElems * FP16_BYTES,
    read_elems: readElems,
    write_elems: writeElems,
    read_breakdown: {
      weight: readWeight * FP16_BYTES,
      activation: readActivation * FP16_BYTES,
    },
    write_breakdown: {
      output: writeOutput * FP16_BYTES,
    },
  }
}

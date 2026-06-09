import type { CycleSnapshot, MemoryAccess } from './types'

const FP16_BYTES = 2

export function formatBytesPerCycle(bytes: number): string {
  return `${bytes} B/cycle`
}

export function deriveMemoryAccess(snapshot: CycleSnapshot): MemoryAccess {
  if (snapshot.memory) {
    return snapshot.memory
  }

  const k = snapshot.progress.k ?? snapshot.pes.length
  const n = snapshot.progress.n ?? snapshot.pes[0]?.length ?? 0
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

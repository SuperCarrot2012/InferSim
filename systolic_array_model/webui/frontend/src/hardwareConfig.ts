export interface HardwarePreset {
  clockGhz: number
  /** Per-PPU SRAM capacity in KiB. */
  ppuSramSizeKb: number
  /** LPDDR peak bandwidth in GB/s. */
  lpddrBandwidthGBps: number
}

export const DEFAULT_HARDWARE_PRESET: HardwarePreset = {
  clockGhz: 1,
  ppuSramSizeKb: 256,
  lpddrBandwidthGBps: 256,
}

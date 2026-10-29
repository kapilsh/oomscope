const KiB = 1024
const MiB = 1024 * 1024
const GiB = 1024 * 1024 * 1024

/** Bytes as the units people actually read memory in. */
export function bytes(n) {
  if (n == null || Number.isNaN(n)) { return '--' }
  const abs = Math.abs(n)
  if (abs >= GiB) { return `${(n / GiB).toFixed(abs >= 10 * GiB ? 1 : 2)} GiB` }
  if (abs >= MiB) { return `${(n / MiB).toFixed(abs >= 10 * MiB ? 1 : 2)} MiB` }
  if (abs >= KiB) { return `${(n / KiB).toFixed(abs >= 10 * KiB ? 0 : 1)} KiB` }
  return `${n} B`
}

/** Compact form for axis ticks, where width is scarce. */
export function bytesTick(n) {
  const abs = Math.abs(n)
  if (abs >= GiB) { return `${+(n / GiB).toFixed(1)}G` }
  if (abs >= MiB) { return `${Math.round(n / MiB)}M` }
  if (abs >= KiB) { return `${Math.round(n / KiB)}K` }
  return `${n}`
}

export function pct(x) {
  if (!Number.isFinite(x)) { return '--' }
  return `${(x * 100).toFixed(x >= 0.1 ? 0 : 1)}%`
}

export function count(n) {
  return n.toLocaleString('en-US')
}

/** Addresses are compared by eye far more often than read, so keep them hex. */
export function addr(a) {
  if (a == null) { return '--' }
  return `0x${a.toString(16)}`
}

/** Microsecond timestamps are only meaningful relative to the trace start. */
export function duration(us) {
  if (us == null || !Number.isFinite(us)) { return '--' }
  if (Math.abs(us) < 1000) { return `${us.toFixed(0)} us` }
  if (Math.abs(us) < 1e6) { return `${(us / 1000).toFixed(1)} ms` }
  return `${(us / 1e6).toFixed(2)} s`
}

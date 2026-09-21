// Turning a raw snapshot dict into the numbers the views render.
//
// The shape we are handed (torch 2.x, `torch.cuda.memory._snapshot()`):
//
//   segments[]      device, address, total_size, allocated_size, active_size,
//                   requested_size, stream, segment_type 'large'|'small',
//                   segment_pool_id, is_expandable, frames[], blocks[]
//   blocks[]        address, size, requested_size,
//                   state 'active_allocated'|'active_pending_free'|'inactive',
//                   frames[]
//   device_traces[] per device, a list of
//                   {action, addr, size, stream, time_us, compile_context, frames[]}
//                   action in alloc | free_requested | free_completed |
//                   segment_alloc | segment_free | oom | snapshot
//
// Two distinctions drive everything downstream, and conflating them is the
// usual reason a memory bug stays mysterious:
//
//   reserved   bytes the caching allocator holds from cudaMalloc. What
//              nvidia-smi bills you for, and what has to fit in the card.
//   allocated  bytes inside those segments currently held by live tensors.
//
// The gap between them is memory you have paid for and cannot use. It is
// either reusable (a free block big enough for the next request) or stranded
// (free, but in pieces too small to satisfy anything) -- the difference
// between "fine" and "OOM at 60% utilisation".

import { blameFrame, frameKey, frameLabel, displayStack } from './frames.js'

export const BlockState = {
  ALLOCATED: 'active_allocated',
  PENDING_FREE: 'active_pending_free',
  INACTIVE: 'inactive',
}

export class SnapshotError extends Error {}

/** Cheap structural check, so a wrong file gives a sentence instead of a stack. */
export function looksLikeSnapshot(raw) {
  return !!raw && typeof raw === 'object' && Array.isArray(raw.segments)
}

/**
 * @param {object} raw decoded snapshot dict
 * @returns {object} normalised model, keyed by device
 */
export function parseSnapshot(raw) {
  if (!looksLikeSnapshot(raw)) {
    throw new SnapshotError(
      'This file decoded, but it is not a memory snapshot: no "segments" list. ' +
      'Expected the output of torch.cuda.memory._dump_snapshot().',
    )
  }

  const traces = Array.isArray(raw.device_traces) ? raw.device_traces : []
  const deviceIds = new Set()
  for (const seg of raw.segments) { deviceIds.add(seg.device ?? 0) }
  traces.forEach((t, i) => { if (t && t.length) { deviceIds.add(i) } })
  if (deviceIds.size === 0) { deviceIds.add(0) }

  const devices = [...deviceIds].sort((a, b) => a - b).map((id) => {
    const segments = raw.segments
      .filter((s) => (s.device ?? 0) === id)
      .map(normaliseSegment)
      .sort((a, b) => a.address - b.address)
    return {
      id,
      segments,
      stats: computeStats(segments),
      timeline: buildTimeline(traces[id] ?? []),
      blame: computeBlame(segments),
    }
  })

  return {
    devices,
    allocatorSettings: raw.allocator_settings ?? null,
    hasAnyTrace: devices.some((d) => d.timeline.hasTrace),
  }
}

function normaliseSegment(seg) {
  const total = seg.total_size ?? 0
  const base = seg.address ?? 0
  let offset = 0
  const blocks = (seg.blocks ?? []).map((b, i) => {
    const size = b.size ?? 0
    // Older snapshots omit per-block addresses; walking the offsets recovers
    // them, since blocks are stored in address order and tile the segment.
    const address = b.address ?? base + offset
    const block = {
      index: i,
      address,
      offset: address - base,
      size,
      requestedSize: b.requested_size ?? size,
      state: b.state ?? BlockState.INACTIVE,
      active: b.state === BlockState.ALLOCATED || b.state === BlockState.PENDING_FREE,
      frames: b.frames ?? [],
    }
    offset += size
    return block
  })

  const blocksTotal = blocks.reduce((n, b) => n + b.size, 0)
  return {
    device: seg.device ?? 0,
    address: base,
    totalSize: total,
    // Trust the blocks over the header: `allocated_size` is a snapshot of the
    // allocator's own counter, and a handful of torch versions disagree with
    // the block list after a pool release.
    allocatedSize: blocks.filter((b) => b.state === BlockState.ALLOCATED).reduce((n, b) => n + b.size, 0),
    activeSize: blocks.filter((b) => b.active).reduce((n, b) => n + b.size, 0),
    requestedSize: blocks.filter((b) => b.active).reduce((n, b) => n + b.requestedSize, 0),
    stream: seg.stream ?? 0,
    type: seg.segment_type ?? 'large',
    poolId: seg.segment_pool_id ?? null,
    isExpandable: !!seg.is_expandable,
    frames: seg.frames ?? [],
    blocks,
    // A segment whose blocks do not add up to its size has a hole the block
    // list does not describe; surface it rather than silently mis-drawing.
    unaccounted: Math.max(0, total - blocksTotal),
  }
}

function computeStats(segments) {
  let reserved = 0, allocated = 0, active = 0, requested = 0
  let freeBytes = 0, largestFree = 0, freeBlocks = 0, activeBlocks = 0
  let small = 0, large = 0, expandable = 0
  const freeSizes = []

  for (const seg of segments) {
    reserved += seg.totalSize
    allocated += seg.allocatedSize
    active += seg.activeSize
    requested += seg.requestedSize
    if (seg.type === 'small') { small++ } else { large++ }
    if (seg.isExpandable) { expandable++ }
    for (const b of seg.blocks) {
      if (b.active) {
        activeBlocks++
      } else {
        freeBlocks++
        freeBytes += b.size
        freeSizes.push(b.size)
        if (b.size > largestFree) { largestFree = b.size }
      }
    }
  }

  // Two different kinds of waste, and they have different fixes.
  //
  // internal: the allocator rounds requests up, so a live block can be bigger
  //   than the tensor in it. Fixed by changing tensor shapes, not by the
  //   allocator.
  // external: free bytes sitting inside reserved segments. Fixed by
  //   expandable_segments, a different max_split_size, or fewer size classes.
  const internalWaste = active - requested
  const externalFree = reserved - active

  return {
    reserved,
    allocated,
    active,
    requested,
    internalWaste,
    externalFree,
    freeBytes,
    largestFreeBlock: largestFree,
    freeBlocks,
    activeBlocks,
    segmentCount: segments.length,
    smallSegments: small,
    largeSegments: large,
    expandableSegments: expandable,
    // Of the memory you are holding but not using, how much is unusable as a
    // single allocation. 1.0 means every free byte is stranded in fragments.
    fragmentation: externalFree > 0 ? 1 - largestFree / externalFree : 0,
    utilisation: reserved > 0 ? active / reserved : 0,
    freeSizes,
  }
}

/**
 * Replay the trace to get memory over time.
 *
 * The trace is a ring buffer capped by `max_entries`, so a long run keeps only
 * the tail and the replay starts mid-flight: running totals can go negative,
 * which just means allocations we never saw are being freed. Rather than
 * clamping (which bends the curve) we shift the whole series up by the deepest
 * excursion and say so, so the shape stays true even when the origin is not.
 */
function buildTimeline(trace) {
  if (!trace || trace.length === 0) {
    return { hasTrace: false, points: [], events: [], oomEvents: [], truncated: false }
  }

  const points = []
  const oomEvents = []
  let allocated = 0, reserved = 0
  let minAllocated = 0, minReserved = 0
  let peakAllocated = 0, peakReserved = 0
  const t0 = trace.find((e) => e.time_us != null)?.time_us ?? 0

  trace.forEach((e, i) => {
    switch (e.action) {
      case 'alloc': allocated += e.size ?? 0; break
      // free_requested fires when the tensor dies, free_completed when the
      // block returns to the pool. Counting both would double every free.
      case 'free_completed': allocated -= e.size ?? 0; break
      case 'segment_alloc': reserved += e.size ?? 0; break
      case 'segment_free': reserved -= e.size ?? 0; break
      case 'oom':
        oomEvents.push({
          index: i,
          size: e.size ?? 0,
          deviceFree: e.device_free ?? null,
          timeUs: e.time_us ?? null,
          frames: e.frames ?? [],
        })
        break
      default: break
    }
    if (allocated < minAllocated) { minAllocated = allocated }
    if (reserved < minReserved) { minReserved = reserved }
    if (allocated > peakAllocated) { peakAllocated = allocated }
    if (reserved > peakReserved) { peakReserved = reserved }
    points.push({
      i,
      t: e.time_us != null ? e.time_us - t0 : null,
      allocated,
      reserved,
      action: e.action,
    })
  })

  const truncated = minAllocated < 0 || minReserved < 0
  if (truncated) {
    const shiftA = -minAllocated
    const shiftR = -minReserved
    for (const p of points) { p.allocated += shiftA; p.reserved += shiftR }
    peakAllocated += shiftA
    peakReserved += shiftR
  }

  return {
    hasTrace: true,
    points,
    events: trace,
    oomEvents,
    truncated,
    peakAllocated,
    peakReserved,
    durationUs: points.length ? (points[points.length - 1].t ?? 0) : 0,
    counts: trace.reduce((acc, e) => {
      acc[e.action] = (acc[e.action] ?? 0) + 1
      return acc
    }, {}),
  }
}

/**
 * Who is holding the memory that is live right now.
 *
 * Grouped over the segment block list rather than the trace, because that is
 * the set of allocations still alive at snapshot time -- which is the set you
 * care about when the next allocation is the one that fails.
 */
function computeBlame(segments) {
  const groups = new Map()
  for (const seg of segments) {
    for (const b of seg.blocks) {
      if (!b.active) { continue }
      const frame = blameFrame(b.frames)
      const key = frameKey(frame)
      let g = groups.get(key)
      if (!g) {
        g = {
          key,
          frame,
          label: frameLabel(frame),
          bytes: 0,
          requested: 0,
          count: 0,
          stack: displayStack(b.frames),
          largest: 0,
        }
        groups.set(key, g)
      }
      g.bytes += b.size
      g.requested += b.requestedSize
      g.count += 1
      if (b.size > g.largest) { g.largest = b.size }
    }
  }
  return [...groups.values()].sort((a, b) => b.bytes - a.bytes)
}

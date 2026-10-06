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
//
// A third distinction only shows up once private pools are in play: which
// pool a segment belongs to. `segment_pool_id` is (0, 0) for the ordinary
// pool; anything else is walled off -- a CUDA graph's working set, a
// torch.cuda.MemPool, a symmetric-memory or NCCL-registered buffer pool. A
// free block in one pool cannot serve an allocation from another, so "free"
// only means something per pool.

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
    const trace = traces[id] ?? []
    const device = {
      id,
      segments,
      stats: computeStats(segments),
      timeline: buildTimeline(trace),
      blame: computeBlame(segments),
      pool: null, // this is the whole device, not one pool of it
      pools: [],
    }
    device.pools = buildPools(device, trace)
    return device
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
    poolKey: poolKey(seg.segment_pool_id),
    isExpandable: !!seg.is_expandable,
    frames: seg.frames ?? [],
    blocks,
    // A segment whose blocks do not add up to its size has a hole the block
    // list does not describe; surface it rather than silently mis-drawing.
    unaccounted: Math.max(0, total - blocksTotal),
    // Trace index of this segment's segment_alloc, or -1 if it predates the
    // trace. Filled in by attributeTrace when there are pools to split.
    allocIndex: -1,
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
function buildTimeline(trace, baseReserved = 0) {
  if (!trace || trace.length === 0) {
    return { hasTrace: false, points: [], events: [], oomEvents: [], truncated: false }
  }

  const points = []
  const oomEvents = []
  let allocated = 0, reserved = baseReserved
  let minAllocated = 0, minReserved = 0
  let peakAllocated = 0, peakReserved = reserved
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

// ---- pools -------------------------------------------------------------

/** "a,b" for a (a, b) pool id; snapshots from before pools existed are all default. */
function poolKey(id) {
  return Array.isArray(id) && id.length === 2 ? `${id[0]},${id[1]}` : '0,0'
}

export const DEFAULT_POOL = '0,0'

/**
 * One view per pool, each shaped like a device (segments, stats, blame,
 * timeline) so every component can render a pool exactly as it renders a
 * device. Empty when there is only the default pool: nothing to choose.
 */
function buildPools(device, trace) {
  const byKey = new Map()
  for (const seg of device.segments) {
    if (!byKey.has(seg.poolKey)) { byKey.set(seg.poolKey, []) }
    byKey.get(seg.poolKey).push(seg)
  }
  // A lone private pool is still worth naming; a lone default pool is not.
  if (byKey.size === 0 || (byKey.size === 1 && byKey.has(DEFAULT_POOL))) { return [] }

  const owner = attributeTrace(trace, device.segments)
  const pools = []
  for (const [key, segments] of byKey) {
    const events = trace.filter((_, i) => owner[i] === key || (key === DEFAULT_POOL && owner[i] === null))
    pools.push({
      id: device.id,
      key,
      poolId: key.split(',').map(Number),
      ...classifyPool(key, segments, events),
      segments,
      stats: computeStats(segments),
      blame: computeBlame(segments),
      // Segments that predate the trace never had a segment_alloc recorded, so
      // their bytes are already reserved when the replay starts.
      timeline: buildTimeline(events, segments
        .filter((sg) => sg.allocIndex < 0)
        .reduce((n, sg) => n + sg.totalSize, 0)),
      // When the default pool is picked out, it also stands in for every event
      // the trace has that no surviving private segment explains.
      attributionNote: key === DEFAULT_POOL
        ? 'Events in segments that were released before the snapshot are counted here: the trace does not say which pool those belonged to.'
        : null,
    })
  }
  pools.sort((a, b) => (a.key === DEFAULT_POOL ? -1 : b.key === DEFAULT_POOL ? 1 : b.stats.reserved - a.stats.reserved))
  for (const p of pools) {
    p.pool = p
    p.pools = pools
  }
  return pools
}

/**
 * Which pool each trace event belongs to, or null when no surviving segment
 * explains it.
 *
 * Trace entries carry an address but no pool id, so the only way to split the
 * history by pool is to place each event inside a segment that still exists at
 * snapshot time -- and only after that segment's own segment_alloc, because an
 * address range can be released and handed out again to a different pool. For
 * a private pool this is close to exact: its segments live as long as its
 * owner does, so they are almost always still there when the snapshot is taken.
 */
function attributeTrace(trace, segments) {
  const owner = new Array(trace.length).fill(null)
  if (trace.length === 0 || segments.length === 0) { return owner }

  const lastAlloc = new Map()
  trace.forEach((e, i) => {
    if (e.action === 'segment_alloc' || e.action === 'segment_map') { lastAlloc.set(e.addr, i) }
  })
  for (const seg of segments) {
    seg.allocIndex = lastAlloc.get(seg.address) ?? -1
  }

  // segments is sorted by address and segments never overlap, so a binary
  // search finds the only candidate.
  trace.forEach((e, i) => {
    if (e.addr == null) { return }
    let lo = 0, hi = segments.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const seg = segments[mid]
      if (e.addr < seg.address) {
        hi = mid - 1
      } else if (e.addr >= seg.address + seg.totalSize) {
        lo = mid + 1
      } else {
        if (i >= seg.allocIndex) { owner[i] = seg.poolKey }
        return
      }
    }
  })
  return owner
}

const isGraphTreesFrame = (f) => /torch\/_inductor\/cudagraph_trees\.py$/.test(f.filename ?? '')
const isSymmMemFrame = (f) => /_symmetric_memory|symm_mem/.test(`${f.filename ?? ''} ${f.name ?? ''}`)

/**
 * What made this pool, as far as the snapshot can tell.
 *
 * The id says some of it. torch numbers a CUDA graph's own private pool
 * (N, 0), and every user-created pool -- graph_pool_handle(), MemPool(), which
 * includes symmetric-memory and NCCL pools -- (0, N). Past that the snapshot is
 * silent: it records no allocator and no owner. So the rest is read off the
 * evidence that is there, and `evidence` says which, so a guess never passes
 * for a fact.
 */
function classifyPool(key, segments, events) {
  const [a, b] = key.split(',').map(Number)
  if (key === DEFAULT_POOL) {
    return {
      kind: 'default',
      label: 'Default pool',
      origin: 'The caching allocator\'s ordinary pool: every allocation not routed anywhere else.',
      evidence: 'pool id (0, 0)',
    }
  }
  if (a > 0 && b === 0) {
    return {
      kind: 'graph',
      label: 'CUDA graph pool',
      origin: 'The private pool of one torch.cuda.graph capture made without pool=. Nothing else can allocate from it.',
      evidence: `pool id (${a}, 0): torch numbers capture-private pools this way`,
    }
  }

  const frames = []
  for (const seg of segments) {
    for (const blk of seg.blocks) { if (blk.active) { frames.push(...blk.frames) } }
  }
  for (const e of events) { if (e.action === 'alloc') { frames.push(...(e.frames ?? [])) } }

  if (frames.some(isGraphTreesFrame)) {
    return {
      kind: 'graph',
      label: 'CUDA graph pool',
      origin: 'Shared by the graphs torch.compile(mode="reduce-overhead") recorded. Cudagraph trees put every graph in one pool so they can reuse each other\'s memory.',
      evidence: 'allocated from torch/_inductor/cudagraph_trees.py',
    }
  }
  if (frames.some(isSymmMemFrame)) {
    return {
      kind: 'symmetric',
      label: 'Symmetric memory pool',
      origin: 'Buffers from torch.distributed._symmetric_memory, mapped so peer GPUs can address them directly.',
      evidence: 'allocated from torch.distributed._symmetric_memory',
    }
  }
  // Graph capture has to run on a side stream, so a user pool whose every
  // segment was made off the default stream is very likely a shared graph
  // pool. Likely, not certain: a MemPool used on a side stream looks the same.
  if (segments.every((s) => s.stream !== 0)) {
    return {
      kind: 'graph',
      label: 'CUDA graph pool',
      origin: 'Most likely shared between graphs via torch.cuda.graph_pool_handle().',
      evidence: `pool id (0, ${b}) with every segment on a side stream, where graph capture runs -- inferred, not recorded`,
    }
  }
  return {
    kind: 'mempool',
    label: 'MemPool',
    origin: 'A torch.cuda.MemPool. Symmetric-memory and NCCL-registered pools (MemPool(allocator, symmetric=True)) look exactly like this in a snapshot.',
    evidence: `pool id (0, ${b}): user-created; the snapshot does not record which allocator backs it`,
  }
}

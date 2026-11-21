import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'

import { addr, bytes, count, duration } from '../lib/format.js'
import { blameFrame, displayStack, frameLabel, provenance } from '../lib/frames.js'
import { buildHistory, layoutAt } from '../lib/history.js'
import { modelsFor } from '../lib/allocTimeline.js'
import { useStore } from '../store.js'
import { Segment } from './Segments.jsx'
import Stack from './Stack.jsx'
import Meta from './Meta.jsx'
import { ForwardStack, ProvenanceRows } from './Provenance.jsx'

const ROW = 20 // px per event in the list

const FILTERS = {
  all: { label: 'every event', test: () => true },
  mem: { label: 'allocs & frees', test: (a) => a === 'alloc' || a === 'free_requested' || a === 'free_completed' },
  seg: { label: 'segment changes', test: (a) => a.startsWith('segment_') },
  oom: { label: 'OOMs & snapshots', test: (a) => a === 'oom' || a === 'snapshot' },
}

const ACTION = {
  alloc: ['alloc', 'alloc'],
  free_requested: ['free req', 'free'],
  free_completed: ['free', 'free'],
  segment_alloc: ['+segment', 'seg'],
  segment_free: ['−segment', 'seg'],
  segment_map: ['+map', 'seg'],
  segment_unmap: ['−unmap', 'seg'],
  oom: ['OOM', 'oom'],
  snapshot: ['snapshot', 'other'],
  annotate: ['annotate', 'other'],
}
const actionOf = (a) => ACTION[a] ?? [a, 'other']

const cache = new WeakMap()
function historyFor(events, segments) {
  let h = cache.get(events)
  if (!h) { h = buildHistory(events, segments); cache.set(events, h) }
  return h
}

/**
 * The allocator's state at any event: torch memory_viz's "Allocator State
 * History". Pick an event on the left (or scrub the strip, or step with the
 * arrow keys) and the segment map shows every segment and block as it stood
 * right after it.
 */
export default function History({ device }) {
  if (!device.timeline.hasTrace) {
    return (
      <div className="card">
        <h3>Allocator state history</h3>
        <p className="sub">
          {device.pool
            ? 'No recorded event landed in this pool, so there is no history to step through — only its end state, on the Segments tab.'
            : 'This snapshot has no trace, so there is only its end state — the Segments tab. Record history before the run to step through how the allocator got there.'}
        </p>
      </div>
    )
  }
  return <HistoryView key={device.pool?.key ?? `cuda:${device.id}`} device={device} />
}

function HistoryView({ device }) {
  const events = device.timeline.events
  const n = events.length
  const h = useMemo(() => historyFor(events, device.segments), [events, device.segments])
  const models = useMemo(() => modelsFor(events, device.segments), [events, device.segments])
  const at = useStore((s) => s.historyAt)
  const setAt = useStore((s) => s.setHistoryAt)
  const t = at == null ? n - 1 : Math.min(n - 1, Math.max(0, at))
  const [filter, setFilter] = useState('all')
  const [selAddr, setSelAddr] = useState(null)
  const mapRef = useRef(null)

  const list = useMemo(() => {
    const f = FILTERS[filter].test
    const out = []
    for (let i = 0; i < n; i++) { if (f(events[i].action)) { out.push(i) } }
    return Int32Array.from(out)
  }, [events, n, filter])

  const streamName = useMemo(() => {
    const names = new Map([[0, 'default stream']])
    for (const s of device.segments) { names.set(s.stream, s.streamName) }
    return (s) => names.get(s) ?? `stream ${addr(s)}`
  }, [device.segments])

  // The map can be tens of thousands of blocks; let it trail the list and the
  // card by a frame rather than hold up the next key press.
  const tMap = useDeferredValue(t)
  const L = useMemo(() => layoutAt(h, tMap, streamName), [h, tMap, streamName])
  const poolOf = useMemo(() => {
    const m = new Map()
    if (!device.pool) { for (const p of device.pools) { m.set(p.key, p) } }
    return m
  }, [device.pool, device.pools])

  // Where t sits in the filtered list: the last listed event at or before it.
  const pos = upperBound(list, t) - 1
  const go = (i) => { if (i >= 0 && i < n) { setAt(i) } }
  const step = (d) => {
    if (list.length === 0) { return }
    const here = pos >= 0 && list[pos] === t
    const k = d < 0 ? (here ? pos + d : pos + d + 1) : pos + d
    go(list[Math.min(list.length - 1, Math.max(0, k))])
  }
  const stepRef = useRef(step)
  useEffect(() => { stepRef.current = step })

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) { return }
      const d = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1, PageDown: 20, PageUp: -20, End: 1e9, Home: -1e9 }[e.key]
      if (d) { e.preventDefault(); stepRef.current(d) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const e = events[t]
  const eMap = events[tMap]
  const mark = eMap && eMap.action !== 'oom' && eMap.action !== 'snapshot' ? eMap.addr : null

  // Bring what the event touched into view, without dragging the page.
  useEffect(() => {
    const box = mapRef.current
    const el = box?.querySelector('.seg.ev, .blocks .ev')?.closest('.seg')
    if (!el || box.scrollHeight <= box.clientHeight) { return }
    const top = el.offsetTop - box.offsetTop
    if (top < box.scrollTop || top + el.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTop = Math.max(0, top - box.clientHeight / 3)
    }
  }, [tMap])

  let selected = null
  if (selAddr != null) {
    for (const s of L.segments) {
      const b = s.blocks.find((x) => x.address === selAddr && x.active)
      if (b) { selected = { segAddress: s.address, index: b.index }; break }
    }
  }
  const onSelect = (sel) => {
    if (!sel) { setSelAddr(null); return }
    const s = L.segments.find((x) => x.address === sel.segAddress)
    setSelAddr(s?.blocks[sel.index]?.address ?? null)
  }

  const t0 = useMemo(() => events.find((x) => x.time_us != null)?.time_us ?? null, [events])
  const S = L.stats

  return (
    <div className="card fill hist">
      <h3>Allocator state at event {count(t)}{t === n - 1 ? ' — the snapshot' : ''}</h3>
      <p className="sub">
        Every segment and block as it stood right after the event picked on the left — torch&apos;s
        {' '}<code>memory_viz</code> &ldquo;allocator state history&rdquo;. Step with the arrow keys, scrub the strip,
        or jump here from an OOM or a pinned band on the Timeline. Rebuilt by undoing the trace from the snapshot.
      </p>

      <Strip models={models} n={n} t={t} ooms={device.timeline.oomEvents} onPick={go} />

      <div className="hist-bar">
        <div className="toggle">
          <span className="lbl">list</span>
          {Object.entries(FILTERS).map(([k, f]) => (
            <button key={k} className={filter === k ? 'on' : ''} onClick={() => setFilter(k)}>{f.label}</button>
          ))}
        </div>
        <span className="muted small num">{count(list.length)} of {count(n)} events</span>
        <span style={{ flex: 1 }} />
        <button className="btn" onClick={() => step(-1)} disabled={pos <= 0 && list[0] >= t} title="previous (↑)">↑ prev</button>
        <button className="btn" onClick={() => step(1)} disabled={pos >= list.length - 1} title="next (↓)">next ↓</button>
        <button className="btn" onClick={() => go(n - 1)} disabled={t === n - 1}>Snapshot</button>
      </div>

      <div className="hist-body">
        <EventList events={events} list={list} t={t} pos={pos} onPick={go} />
        <div className="hist-main">
          <EventCard e={e} t={t} t0={t0} stats={S} streamName={streamName} />
          <div className="hist-stats num">
            <span><b>{bytes(S.reserved)}</b> reserved</span>
            <span><b>{bytes(S.allocated)}</b> allocated</span>
            <span><b>{bytes(S.freeBytes)}</b> free in {count(S.freeBlocks)} piece{S.freeBlocks === 1 ? '' : 's'}, largest <b>{bytes(S.largestFree)}</b></span>
            <span>{count(S.segments)} segment{S.segments === 1 ? '' : 's'} · {count(S.liveBlocks)} live block{S.liveBlocks === 1 ? '' : 's'}</span>
          </div>
          {h.inconsistent > 0 && (
            <p className="note warn">
              {count(h.inconsistent)} event{h.inconsistent === 1 ? '' : 's'} could not be reconciled with the snapshot
              (an alloc or free the trace lost), so some states here are approximate.
            </p>
          )}
          <div className="hist-map" ref={mapRef}>
            {L.segments.length === 0 && <p className="muted">Nothing reserved at this point.</p>}
            {L.segments.map((seg) => (
              <Segment
                key={`${seg.address}:${seg.madeAt}`}
                seg={seg}
                pool={poolOf.get(seg.poolKey)}
                selected={selected}
                onSelect={onSelect}
                mark={mark}
                extra={(b) => <Lifetime b={b} n={n} onGo={go} />}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

/** The selected event, and what the allocator looked like right after it. */
function EventCard({ e, t, t0, stats, streamName }) {
  if (!e) { return null }
  const [label] = actionOf(e.action)
  const frame = blameFrame(e.frames)
  const prov = provenance(e)
  const meta = typeof e.user_metadata === 'string' ? e.user_metadata : ''
  const oom = e.action === 'oom'
  return (
    <div className={`detail hev-card${oom ? ' oom' : ''}`}>
      <div className="row">
        <span className="mono">#{count(t)}</span>
        <b>{label}</b>
        {e.size != null && <b>{bytes(e.size)}</b>}
        {!oom && e.addr != null && <span className="mono">{addr(e.addr)}</span>}
        {e.stream != null && e.stream !== 0 && <span>{streamName(e.stream)}</span>}
        {e.time_us != null && t0 != null && <span className="muted">at {duration(e.time_us - t0)}</span>}
        {meta && <Meta text={meta} clip={60} />}
      </div>
      {oom && (
        <div className="row">
          <span>
            {e.device_free != null && <>the device had {bytes(e.device_free)} free outside the cache · </>}
            {stats.largestFree >= (e.size ?? 0)
              ? <>a free block of {bytes(stats.largestFree)} existed, but in another pool or on another stream</>
              : stats.freeBytes >= (e.size ?? 0)
                ? <><b>fragmentation</b>: {bytes(stats.freeBytes)} was free in the cache, but no single block was big enough (largest {bytes(stats.largestFree)})</>
                : <>the cache held only {bytes(stats.freeBytes)} free, so the allocator had to ask the driver — and the card was full</>}
          </span>
        </div>
      )}
      {frame && <div className="row"><span>at <b className="mono">{frameLabel(frame)}</b></span></div>}
      <ProvenanceRows p={prov} frames={e.frames} />
      {e.frames?.length > 0 && (
        <details className="fwd">
          <summary>stack</summary>
          <Stack frames={displayStack(e.frames)} limit={40} />
        </details>
      )}
      <ForwardStack frames={prov.forward} />
    </div>
  )
}

/** When the selected block was made and freed, as links to those moments. */
function Lifetime({ b, n, onGo }) {
  return (
    <div className="row">
      <span>
        {b.start < 0 ? 'made before the trace began' : <>made at <button className="linkish" onClick={() => onGo(b.start)}>event {count(b.start)}</button></>}
        {' · '}
        {b.end >= n ? 'still alive at the snapshot' : <>freed at <button className="linkish" onClick={() => onGo(b.end)}>event {count(b.end)}</button></>}
        {!b.sizeExact && <> · footprint estimated: the trace records only the {bytes(b.requestedSize)} asked for</>}
        {b.framesFromFree && <> · the stack shown was recorded on its free</>}
      </span>
    </div>
  )
}

/** The trace as a virtualised list: a million rows cost what twenty do. */
function EventList({ events, list, t, pos, onPick }) {
  const ref = useRef(null)
  const [scroll, setScroll] = useState(0)
  const [height, setHeight] = useState(400)

  useEffect(() => {
    const el = ref.current
    if (!el) { return }
    const ro = new ResizeObserver(([x]) => setHeight(x.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Keep the current event in view as it moves.
  useEffect(() => {
    const el = ref.current
    if (!el || pos < 0) { return }
    const y = pos * ROW
    if (y < el.scrollTop || y + ROW > el.scrollTop + el.clientHeight) {
      el.scrollTop = Math.max(0, y - el.clientHeight / 2)
    }
  }, [pos])

  const first = Math.max(0, Math.floor(scroll / ROW) - 10)
  const last = Math.min(list.length, first + Math.ceil(height / ROW) + 20)
  const rows = []
  for (let k = first; k < last; k++) {
    const i = list[k]
    const e = events[i]
    const [label, kind] = actionOf(e.action)
    rows.push(
      <button
        key={i}
        className={`ev a-${kind}${i === t ? ' on' : ''}`}
        style={{ top: k * ROW }}
        onClick={() => onPick(i)}
      >
        <span className="i">{i}</span>
        <span className="a">{label}</span>
        <span className="s">{e.size != null ? bytes(e.size) : ''}</span>
        <span className="ad">{e.action === 'oom' ? '' : e.addr != null ? addr(e.addr) : ''}</span>
      </button>,
    )
  }
  return (
    <div className="hev mono" ref={ref} onScroll={(e) => setScroll(e.currentTarget.scrollTop)}>
      <div style={{ height: list.length * ROW, position: 'relative' }}>{rows}</div>
      {list.length === 0 && <p className="muted" style={{ padding: 10 }}>No events of this kind.</p>}
    </div>
  )
}

/**
 * Allocated and reserved over the whole trace, one column per pixel, as a
 * scrubber: drag along it to move through time.
 */
function Strip({ models, n, t, ooms, onPick }) {
  const ref = useRef(null)
  const canvasRef = useRef(null)
  const [w, setW] = useState(0)
  const H = 40

  useEffect(() => {
    const el = ref.current
    if (!el) { return }
    const ro = new ResizeObserver(([x]) => setW(Math.floor(x.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const c = canvasRef.current
    if (!c || w === 0) { return }
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const W = Math.round(w * dpr), Hd = Math.round(H * dpr)
    c.width = W; c.height = Hd
    const ctx = c.getContext('2d')
    const css = getComputedStyle(document.documentElement)
    const alc = models.blocks.totals, res = models.segments.totals
    const yMax = Math.max(models.blocks.peak, models.segments.peak, 1) * 1.05
    const colA = new Float64Array(W), colR = new Float64Array(W)
    for (let i = 0; i < n; i++) {
      const col = Math.min(W - 1, Math.floor((i / n) * W))
      if (alc[i] > colA[col]) { colA[col] = alc[i] }
      if (res[i] > colR[col]) { colR[col] = res[i] }
    }
    // Columns narrower than an event carry the last value forward.
    for (let col = 1; col < W; col++) {
      if (colA[col] === 0 && colR[col] === 0) { colA[col] = colA[col - 1]; colR[col] = colR[col - 1] }
    }
    const yOf = (v) => Hd - (v / yMax) * Hd
    ctx.clearRect(0, 0, W, Hd)
    ctx.fillStyle = css.getPropertyValue('--blk-active').trim() || '#76b900'
    ctx.globalAlpha = 0.55
    for (let col = 0; col < W; col++) { const y = yOf(colA[col]); ctx.fillRect(col, y, 1, Hd - y) }
    ctx.globalAlpha = 1
    ctx.strokeStyle = css.getPropertyValue('--line-reserved').trim() || '#3987e5'
    ctx.lineWidth = dpr
    ctx.beginPath()
    for (let col = 0; col < W; col++) { const y = yOf(colR[col]); if (col === 0) { ctx.moveTo(0, y) } else { ctx.lineTo(col, y) } }
    ctx.stroke()
  }, [models, n, w])

  const pick = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    onPick(Math.min(n - 1, Math.floor(f * n)))
  }
  const xOf = (i) => ((i + 0.5) / n) * 100

  return (
    <div
      className="hist-strip"
      ref={ref}
      style={{ height: H }}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); pick(e) }}
      onPointerMove={(e) => { if (e.buttons & 1) { pick(e) } }}
      title="allocated (filled) and reserved (line) over the trace — click or drag to move"
    >
      <canvas ref={canvasRef} style={{ width: '100%', height: H }} />
      {ooms.map((o) => <i key={o.index} className="oom" style={{ left: `${xOf(o.index)}%` }} />)}
      <i className="cur" style={{ left: `${xOf(t)}%` }} />
    </div>
  )
}

function upperBound(arr, v) {
  let lo = 0, hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] <= v) { lo = mid + 1 } else { hi = mid }
  }
  return lo
}

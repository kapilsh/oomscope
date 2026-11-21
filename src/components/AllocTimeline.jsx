import { useEffect, useMemo, useRef, useState } from 'react'

import { addr, bytes, bytesTick, count, duration } from '../lib/format.js'
import { blameFrame, displayStack, frameKey, frameLabel, fxFrame, fxLabel, provenance } from '../lib/frames.js'
import { describe, modelsFor, rangeMax, rasterize, slotColors } from '../lib/allocTimeline.js'
import { useStore } from '../store.js'
import Stack from './Stack.jsx'
import Meta from './Meta.jsx'
import { ForwardStack, ProvenanceRows } from './Provenance.jsx'

const PAD = { l: 62, r: 14, t: 12, b: 26 }
// Same breakpoint as the fit-to-window CSS: below it the page just scrolls.
const FIT_QUERY = '(min-width: 900px) and (min-height: 560px)'
// Legend margin, card padding and page gutter under the chart.
const BELOW_CHART = 10 + 16 + 14
const MIN_SPAN = 16 // events; narrower than this and a column is mostly one event


const sourceKey = (frames) => frameKey(blameFrame(frames))

/**
 * Every allocation over the trace, stacked: torch memory_viz's active memory
 * timeline. Drawn to a canvas from a per-pixel rasteriser (see
 * lib/allocTimeline.js), so the cost of a frame is the pixels on screen, not
 * the number of allocations behind them.
 */
export default function AllocTimeline({ device, children }) {
  const tl = device.timeline
  const [mode, setMode] = useState('blocks')
  const [colorBy, setColorBy] = useState('alloc')
  const [tipStack, setTipStack] = useState(false) // hover shows the full stack, not one line
  const models = useMemo(() => modelsFor(tl.events, device.segments), [tl.events, device.segments])
  const m = models[mode]
  const overlay = models[mode === 'blocks' ? 'segments' : 'blocks'].totals
  const n = m.n
  const colors = useMemo(() => slotColors(m, colorBy, sourceKey), [m, colorBy])

  const [view, setView] = useState({ x0: 0, x1: n })
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [frame, setFrame] = useState(null)
  const [hover, setHover] = useState(null) // { c, slot, px, py }
  const [pinned, setPinned] = useState(null) // slot
  const [brush, setBrush] = useState(null) // { from, to } css px
  const showHistory = useStore((s) => s.showHistory)

  const wrapRef = useRef(null)
  const legendRef = useRef(null)
  const detailRef = useRef(null)
  const canvasRef = useRef(null)
  const bufRef = useRef(null)
  const imgRef = useRef(null)
  const viewRef = useRef(view)
  useEffect(() => { viewRef.current = view }, [view])
  const dragRef = useRef(null)

  const width = size.w
  const H = size.h
  const plotW = Math.max(1, width - PAD.l - PAD.r)
  const plotH = Math.max(1, H - PAD.t - PAD.b)

  const streamNames = useMemo(() => {
    const s = new Map([[0, 'default stream']])
    for (const seg of device.segments) { s.set(seg.stream, seg.streamName) }
    return s
  }, [device.segments])

  // The chart takes the card's width and, through CSS, most of the viewport's
  // height; both are read back so the raster matches the box exactly.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) { return }
    const ro = new ResizeObserver(([e]) => {
      const w = Math.floor(e.contentRect.width), h = Math.floor(e.contentRect.height)
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // On a screen with room, the chart is exactly as tall as the window allows
  // below everything above it, with the legend still in view: the whole graph
  // on screen, and anything pinned under it a scroll away.
  const [fitH, setFitH] = useState(null)
  useEffect(() => {
    const mq = window.matchMedia(FIT_QUERY)
    const measure = () => {
      const el = wrapRef.current
      if (!el || !mq.matches) { setFitH(null); return }
      const top = el.getBoundingClientRect().top + window.scrollY
      const below = (legendRef.current?.offsetHeight ?? 24) + BELOW_CHART
      setFitH(Math.max(300, Math.floor(window.innerHeight - top - below)))
    }
    const id = requestAnimationFrame(measure)
    window.addEventListener('resize', measure)
    return () => { cancelAnimationFrame(id); window.removeEventListener('resize', measure) }
  }, [width, mode])

  const clamp = (x0, x1) => clampView(n, x0, x1)

  // Zoom on ctrl/cmd-scroll and pan on shift- or sideways scroll. A plain
  // vertical scroll is left to the page, so the chart never traps it.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) { return }
    const onWheel = (e) => {
      const { x0, x1 } = viewRef.current
      const span = x1 - x0
      const rect = el.getBoundingClientRect()
      const frac = Math.min(1, Math.max(0, (e.clientX - rect.left - PAD.l) / Math.max(1, rect.width - PAD.l - PAD.r)))
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        const k = Math.exp(e.deltaY * 0.0025)
        const at = x0 + frac * span
        setView(clampView(n, at - frac * span * k, at + (1 - frac) * span * k))
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault()
        const d = (e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX) / Math.max(1, rect.width - PAD.l - PAD.r) * span
        setView(clampView(n, x0 + d, x1 + d))
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [n])

  // Rasterise on the next frame, so a burst of wheel events costs one draw.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width === 0) { return }
    const id = requestAnimationFrame(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const W = Math.max(1, Math.round(plotW * dpr))
      const Hd = Math.max(1, Math.round(plotH * dpr))
      const a = Math.floor(view.x0), b = Math.ceil(view.x1)
      const yMax = niceCeil(Math.max(rangeMax(m.totals, a, b), rangeMax(overlay, a, b), 1))
      bufRef.current = rasterize(m, {
        x0: view.x0, x1: view.x1, width: W, height: Hd, yMax, colors, overlay, buf: bufRef.current,
      })
      if (canvas.width !== W || canvas.height !== Hd) { canvas.width = W; canvas.height = Hd }
      setFrame({ yMax, W, H: Hd, x0: view.x0, x1: view.x1, buf: bufRef.current })
    })
    return () => cancelAnimationFrame(id)
  }, [m, overlay, colors, view.x0, view.x1, width, plotW, plotH])

  // Copy the raster to the canvas, lifting the hovered and pinned bands.
  const hoverSlot = hover?.slot ?? null
  useEffect(() => {
    const canvas = canvasRef.current
    if (!frame || !canvas) { return }
    const ctx = canvas.getContext('2d')
    let img = imgRef.current
    if (!img || img.width !== frame.W || img.height !== frame.H) {
      img = imgRef.current = ctx.createImageData(frame.W, frame.H)
    }
    const out = new Uint32Array(img.data.buffer)
    const { pix, ids } = frame.buf
    out.set(pix)
    // A pinned band can be a sliver, so everything else steps back while one is.
    const h = hoverSlot == null ? -1 : hoverSlot + 1
    const p = pinned == null ? -1 : pinned + 1
    if (p > 0) {
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]
        if (id === p) { out[i] = lighten(out[i], 0.3) } else if (id === h) { out[i] = lighten(out[i], 0.1) } else if (id > 0) { out[i] = shade(out[i], 0.38) }
      }
    } else if (h > 0) {
      for (let i = 0; i < ids.length; i++) {
        if (ids[i] === h) { out[i] = lighten(out[i], 0.45) }
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [frame, hoverSlot, pinned])

  const y = (v) => PAD.t + (1 - v / (frame?.yMax || 1)) * plotH
  const xOfEvent = (i) => PAD.l + ((i - view.x0) / (view.x1 - view.x0)) * plotW

  const overlayPath = useMemo(() => {
    if (!frame) { return '' }
    const { W, yMax, buf } = frame
    const { overlayMax, sample } = buf
    let d = ''
    let last = null
    for (let c = 0; c < W; c++) {
      if (sample[c] < 0) { break }
      const yy = (PAD.t + (1 - overlayMax[c] / yMax) * plotH).toFixed(1)
      const xx = (PAD.l + (c / W) * plotW).toFixed(1)
      if (last === null) { d += `M${xx},${yy}` } else if (yy !== last) { d += `H${xx}V${yy}` }
      last = yy
    }
    if (last !== null) {
      let end = W
      while (end > 0 && sample[end - 1] < 0) { end-- }
      d += `H${(PAD.l + (end / W) * plotW).toFixed(1)}`
    }
    return d
  }, [frame, plotW, plotH])

  const xTicks = useMemo(() => eventTicks(view.x0, view.x1), [view.x0, view.x1])
  const yTickVals = frame ? yTicks(frame.yMax) : []

  const locate = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return { px: e.clientX - rect.left, py: e.clientY - rect.top }
  }
  const pick = (px, py) => {
    if (!frame) { return null }
    const c = Math.floor(((px - PAD.l) / plotW) * frame.W)
    const r = Math.floor(((py - PAD.t) / plotH) * frame.H)
    if (c < 0 || c >= frame.W || frame.buf.sample[c] < 0) { return null }
    const id = r >= 0 && r < frame.H ? frame.buf.ids[r * frame.W + c] : 0
    return { c, slot: id > 0 ? id - 1 : null, px, py }
  }

  const onPointerDown = (e) => {
    if (e.button !== 0) { return }
    const { px } = locate(e)
    dragRef.current = { from: px, moved: false }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e) => {
    const { px, py } = locate(e)
    const d = dragRef.current
    if (d) {
      if (Math.abs(px - d.from) > 3) { d.moved = true }
      if (d.moved) { setBrush({ from: d.from, to: px }) }
    }
    const h = pick(px, py)
    // Re-render only when the column or the band changes, unless a tooltip
    // has to follow the pointer.
    if (h?.c !== hover?.c || h?.slot !== hover?.slot || (h?.slot != null && !d)) { setHover(h) }
  }
  const onPointerUp = (e) => {
    const d = dragRef.current
    dragRef.current = null
    setBrush(null)
    if (!d) { return }
    const { px, py } = locate(e)
    if (d.moved) {
      const toEvent = (x) => view.x0 + ((Math.min(Math.max(x, PAD.l), PAD.l + plotW) - PAD.l) / plotW) * (view.x1 - view.x0)
      const a = toEvent(Math.min(d.from, px)), b = toEvent(Math.max(d.from, px))
      if (b - a > 0) { setView(clamp(a, b)) }
    } else {
      const h = pick(px, py)
      setPinned(h?.slot != null && h.slot !== pinned ? h.slot : null)
    }
  }

  const switchMode = (next) => {
    if (next === mode) { return }
    setMode(next)
    setHover(null)
    setPinned(null)
  }

  const zoomed = view.x0 > 0 || view.x1 < n
  const hoverEvent = hover && frame ? frame.buf.sample[hover.c] : null
  const hovered = hover?.slot != null && !brush ? describe(m, hover.slot) : null
  const pin = pinned != null ? describe(m, pinned) : null
  const t0 = useMemo(() => tl.events.find((e) => e.time_us != null)?.time_us ?? null, [tl.events])

  const what = mode === 'blocks' ? 'allocation' : 'segment'
  const overlayName = mode === 'blocks' ? 'reserved' : 'allocated'

  const zoomTo = (a, b, minPad = MIN_SPAN) => {
    const pad = Math.max(minPad, (b - a) * 0.15)
    setView(clamp(a - pad, b + pad))
  }

  return (
    <div className="card">
      <h3>Active {mode === 'blocks' ? 'allocations' : 'segments'} over the trace</h3>
      <p className="sub">
        {mode === 'blocks'
          ? <>Every allocation as a band as tall as its size, stacked in the order it was made — torch&apos;s own <code>memory_viz</code> view. A freed band leaves a gap the ones above slide down into, so the top of the stack is allocated memory and the line above it is what the allocator has reserved.</>
          : <>Every segment the allocator took from <code>cudaMalloc</code>, stacked the same way: the top of the stack is reserved memory, and the line under it is how much of that live tensors were using.</>}
      </p>
      <p className="sub">
        {count(n)} events{tl.durationUs ? ` spanning ${duration(tl.durationUs)}` : ''} · {count(m.count)} {what}s
        {m.base > 0 && <> ({bytes(m.base)} made before the trace began)</>}
        {' '}· peak allocated <b>{bytes(models.blocks.peak)}</b> · peak reserved <b>{bytes(models.segments.peak)}</b>
      </p>

      <div className="atl-bar">
        <div className="toggle">
          <button className={mode === 'blocks' ? 'on' : ''} onClick={() => switchMode('blocks')}>allocations</button>
          <button className={mode === 'segments' ? 'on' : ''} onClick={() => switchMode('segments')}>segments</button>
        </div>
        <div className="toggle">
          <span className="lbl">colour by</span>
          <button className={colorBy === 'alloc' ? 'on' : ''} onClick={() => setColorBy('alloc')}>{what}</button>
          <button className={colorBy === 'source' ? 'on' : ''} onClick={() => setColorBy('source')}>source line</button>
        </div>
        <div className="toggle">
          <span className="lbl">hover shows</span>
          <button className={!tipStack ? 'on' : ''} onClick={() => setTipStack(false)}>source line</button>
          <button className={tipStack ? 'on' : ''} onClick={() => setTipStack(true)}>full stack</button>
        </div>
        <span style={{ flex: 1 }} />
        {zoomed && (
          <>
            <span className="muted num small">events {count(Math.floor(view.x0))}–{count(Math.ceil(view.x1))} of {count(n)}</span>
            <button className="btn" onClick={() => setView({ x0: 0, x1: n })}>Reset zoom</button>
          </>
        )}
      </div>

      <div className="atl" ref={wrapRef} style={fitH ? { height: fitH } : undefined}>
        {width > 0 && (
          <svg className="chart" width={width} height={H} aria-hidden="true">
            {yTickVals.map((t) => (
              <line key={t} className="grid" x1={PAD.l} x2={width - PAD.r} y1={y(t)} y2={y(t)} />
            ))}
          </svg>
        )}
        <canvas
          ref={canvasRef}
          style={{ left: PAD.l, top: PAD.t, width: plotW, height: plotH }}
        />
        {width > 0 && (
          <svg
            className="chart over"
            width={width}
            height={H}
            viewBox={`0 0 ${width} ${H}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => { if (!dragRef.current) { setHover(null) } }}
            onDoubleClick={() => setView({ x0: 0, x1: n })}
          >
            {yTickVals.map((t) => (
              <text key={t} className="axis" x={PAD.l - 8} y={y(t) + 3} textAnchor="end">{bytesTick(t)}</text>
            ))}
            {xTicks.map((t) => {
              const xx = xOfEvent(t)
              return xx >= PAD.l - 1 && xx <= width - PAD.r + 1 && (
                <text key={t} className="axis" x={xx} y={H - 8} textAnchor="middle">{count(t)}</text>
              )
            })}

            <path className={mode === 'blocks' ? 'ovl res-line' : 'ovl alc-line'} d={overlayPath} />

            {tl.oomEvents.filter((o) => o.index >= view.x0 && o.index < view.x1).map((o) => (
              <line key={o.index} className="oom" x1={xOfEvent(o.index + 0.5)} x2={xOfEvent(o.index + 0.5)} y1={PAD.t} y2={PAD.t + plotH} />
            ))}

            {pin && <Lifetime d={pin} xOf={xOfEvent} view={view} n={n} plotH={plotH} />}

            {hover && frame && (
              <line
                className="cursor"
                x1={PAD.l + ((hover.c + 0.5) / frame.W) * plotW}
                x2={PAD.l + ((hover.c + 0.5) / frame.W) * plotW}
                y1={PAD.t}
                y2={PAD.t + plotH}
              />
            )}
            {brush && (
              <rect
                className="brush"
                x={Math.max(PAD.l, Math.min(brush.from, brush.to))}
                width={Math.abs(brush.to - brush.from)}
                y={PAD.t}
                height={plotH}
              />
            )}
          </svg>
        )}
        {hovered && <Tip d={hovered} what={what} full={tipStack} at={hover} width={width} height={H} />}
      </div>

      <div className="legend" ref={legendRef}>
        <span><i style={{ background: mode === 'blocks' ? 'var(--line-reserved)' : 'var(--line-allocated)' }} />{overlayName}</span>
        {tl.oomEvents.length > 0 && <span><i style={{ background: 'var(--line-oom)' }} />OOM</span>}
        <span className="spacer" style={{ flex: 1 }} />
        {hover && frame ? (
          <span className="num">
            event {count(hoverEvent)} · stack {bytes(frame.buf.stackTop[hover.c])} · {overlayName} {bytes(overlay[hoverEvent])}
          </span>
        ) : pin ? (
          <button className="linkish" onClick={() => detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
            pinned {bytes(pin.size)} {what} — details below ↓
          </button>
        ) : (
          <span className="muted">drag to zoom · ctrl/⌘-scroll to zoom · shift-scroll to pan · double-click to reset · click a band to pin it</span>
        )}
      </div>

      {/* Under the chart: the chart is sized to fill the window on its own, so
          what is pinned below it only ever scrolls the page, never squeezes it. */}
      {pin && (
        <div ref={detailRef}>
          <Detail
            d={pin}
            what={what}
            t0={t0}
            n={n}
            streamNames={streamNames}
            onZoom={() => zoomTo(Math.max(0, pin.start), Math.min(n, pin.end + 1))}
            onHistory={showHistory}
            onClose={() => setPinned(null)}
          />
        </div>
      )}
      {tl.oomEvents.length > 0 && (
        <OomList events={tl.oomEvents} onShow={(i) => zoomTo(i, i + 1, Math.max(MIN_SPAN, n * 0.02))} onHistory={showHistory} />
      )}
      {children}
    </div>
  )
}

/**
 * The allocation that failed, and where it was called from. This is the
 * request the allocator could not satisfy -- not necessarily the code that
 * wasted the memory.
 */
function OomList({ events, onShow, onHistory }) {
  return (
    <div className="oomlist">
      <h4>Out-of-memory events</h4>
      <p className="muted small">
        The request the allocator could not satisfy — not necessarily the code that wasted the memory.
      </p>
      {events.map((o, i) => {
        const frame = blameFrame(o.frames)
        const prov = provenance(o.event)
        return (
          <div key={i} className="detail">
            <div className="row">
              <span>asked for <b>{bytes(o.size)}</b></span>
              {o.deviceFree != null && <><span>·</span><span>{bytes(o.deviceFree)} free on device</span></>}
              <span>·</span><span>event #{count(o.index)}</span>
              <span style={{ flex: 1 }} />
              <button className="linkish" onClick={() => onShow(o.index)}>show on chart</button>
              <button className="linkish" onClick={() => onHistory(o.index)} title="The segment map as it stood when this request failed: what was free, and in what pieces">allocator state here</button>
            </div>
            {frame && <div className="row"><span>at <b className="mono">{frameLabel(frame)}</b></span></div>}
            <ProvenanceRows p={prov} frames={o.frames} />
            <Stack frames={displayStack(o.frames)} limit={25} />
            <ForwardStack frames={prov.forward} />
          </div>
        )
      })}
    </div>
  )
}

/**
 * What is under the pointer, next to it: the line that made it, or with
 * "full stack" the readable stack -- memory_viz's hover, without having to
 * look away from the chart.
 */
function Tip({ d, what, full, at, width, height }) {
  const frame = blameFrame(d.frames)
  const fx = fxLabel(fxFrame(d.frames))
  const style = {}
  if (at.px > width * 0.55) { style.right = width - at.px + 14 } else { style.left = at.px + 14 }
  if (at.py > height * 0.5) { style.bottom = height - at.py + 10 } else { style.top = at.py + 10 }
  return (
    <div className={`atl-tip${full ? ' full' : ''}`} style={style}>
      <div className="hd">
        <b>{bytes(d.size)}</b> {what}
        <span className="muted"> · {d.start < 0 ? 'before the trace' : `event ${count(d.start)}`} → {d.alive ? 'snapshot' : `event ${count(d.end)}`}</span>
        {d.prov.compileContext && <code className="ctx">{d.prov.compileContext}</code>}
      </div>
      {d.metadata && <div><Meta text={d.metadata} clip={60} /></div>}
      {full
        ? <Stack frames={displayStack(d.frames)} limit={30} />
        : (
          <>
            <div className="mono line">{frame ? frameLabel(frame) : 'no stack recorded'}</div>
            {fx && <div className="mono line fx">FX {fx}</div>}
          </>
        )}
      {d.prov.forward.length > 0 && <div className="muted">+ forward-pass stack — click to pin</div>}
    </div>
  )
}

/** Where the pinned band starts and ends, marked on the axis. */
function Lifetime({ d, xOf, view, n, plotH }) {
  const a = d.start < 0 ? view.x0 : d.start + 0.5
  const b = d.end >= n ? view.x1 : d.end + 0.5
  if (b < view.x0 || a > view.x1) { return null }
  const x0 = Math.max(xOf(Math.max(a, view.x0)), xOf(view.x0))
  const x1 = Math.min(xOf(Math.min(b, view.x1)), xOf(view.x1))
  return <rect className="life" x={x0} width={Math.max(1, x1 - x0)} y={PAD.t + plotH} height={3} />
}

function Detail({ d, what, t0, n, streamNames, onZoom, onHistory, onClose }) {
  const frame = blameFrame(d.frames)
  const at = (i, us) => `event ${count(i)}${us != null && t0 != null ? ` (${duration(us - t0)})` : ''}`
  return (
    <div className="detail">
      <div className="row">
        <span><b>{bytes(d.size)}</b> {what}</span>
        {d.address != null && <><span>·</span><span className="mono">{addr(d.address)}</span></>}
        {d.stream != null && <><span>·</span><span>{streamNames.get(d.stream) ?? `stream ${addr(d.stream)}`}</span></>}
        {d.metadata && <><span>·</span><Meta text={d.metadata} /></>}
        <span style={{ flex: 1 }} />
        <button className="linkish" onClick={onZoom}>zoom to its lifetime</button>
        {d.start >= 0 && <button className="linkish" onClick={() => onHistory(d.start)} title="The segment map right after this was allocated">allocator state when made</button>}
        {d.end < n && <button className="linkish" onClick={() => onHistory(Math.max(0, d.end - 1))} title="The segment map just before this was freed">…before its free</button>}
        <button className="linkish" onClick={onClose}>close</button>
      </div>
      <div className="row">
        <span>
          {d.start < 0 ? 'made before the trace began' : <>made at {at(d.start, d.startUs)}</>}
          {' · '}
          {d.alive ? 'still alive at the snapshot' : <>freed at {at(d.end, d.endUs)}</>}
          {d.start >= 0 && !d.alive && <> · lived {count(d.end - d.start)} events{d.startUs != null && d.endUs != null ? `, ${duration(d.endUs - d.startUs)}` : ''}</>}
          {d.start < 0 && d.end < n && <> · the stack shown was recorded on its free</>}
        </span>
      </div>
      {frame && <div className="row"><span>at <b className="mono">{frameLabel(frame)}</b></span></div>}
      <ProvenanceRows p={d.prov} frames={d.frames} />
      <Stack frames={displayStack(d.frames)} limit={25} />
      <ForwardStack frames={d.prov.forward} />
    </div>
  )
}

/** Keep a view inside the trace and no narrower than MIN_SPAN events. */
function clampView(n, x0, x1) {
  const span = Math.min(n, Math.max(Math.min(MIN_SPAN, n), x1 - x0))
  const lo = Math.min(Math.max(0, x0), n - span)
  return { x0: lo, x1: lo + span }
}

function lighten(c, k) {
  const r = c & 0xff, g = (c >>> 8) & 0xff, b = (c >>> 16) & 0xff
  const f = (v) => (v + (255 - v) * k) | 0
  return ((255 << 24) | (f(b) << 16) | (f(g) << 8) | f(r)) >>> 0
}

function shade(c, k) {
  const r = c & 0xff, g = (c >>> 8) & 0xff, b = (c >>> 16) & 0xff
  return ((255 << 24) | (((b * k) | 0) << 16) | (((g * k) | 0) << 8) | ((r * k) | 0)) >>> 0
}

function niceStep(raw) {
  const pow = 2 ** Math.floor(Math.log2(Math.max(raw, 1)))
  for (const k of [1, 1.5, 2, 3, 4, 6, 8]) {
    if (pow * k >= raw) { return pow * k }
  }
  return pow * 8
}

/** A little headroom, so the highest band never touches the frame. */
function niceCeil(max) {
  return max * 1.05
}

function yTicks(max) {
  const step = niceStep(max / 4)
  const out = []
  for (let v = 0; v <= max * 1.0001; v += step) { out.push(v) }
  return out
}

function eventTicks(x0, x1) {
  const span = x1 - x0
  const raw = span / 6
  const pow = 10 ** Math.floor(Math.log10(Math.max(raw, 1)))
  const step = [1, 2, 5, 10].map((k) => k * pow).find((s) => s >= raw) ?? 10 * pow
  const out = []
  for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) { out.push(v) }
  return out
}

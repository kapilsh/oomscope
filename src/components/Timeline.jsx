import { useMemo, useState } from 'react'

import { bytes, bytesTick, count, duration } from '../lib/format.js'
import { blameFrame, frameLabel, displayStack } from '../lib/frames.js'
import Stack from './Stack.jsx'

const W = 1000
const H = 300
const PAD = { l: 62, r: 14, t: 12, b: 26 }

/**
 * Reserved and allocated over the recorded trace.
 *
 * The gap between the two curves is the whole story of a caching allocator: it
 * grows when a freed block cannot be reused and the allocator takes more from
 * the driver instead. A step up in reserved that never comes back down, while
 * allocated stays flat, is fragmentation happening in front of you.
 */
export default function Timeline({ device }) {
  const tl = device.timeline
  const [hover, setHover] = useState(null)

  // One point per pixel column is plenty; a 4k-event trace does not need 4k
  // path segments, and downsampling by max keeps the peaks that matter.
  const series = useMemo(() => downsample(tl.points, W - PAD.l - PAD.r), [tl.points])

  if (!tl.hasTrace) {
    return (
      <div className="card">
        <h3>Timeline</h3>
        <p className="sub">
          This snapshot has no trace. The segment view still works — it is the end state — but to
          see memory over time, record history before the run:
        </p>
        <pre className="mono" style={{ margin: 0, color: 'var(--text-dim)', fontSize: 12.5, lineHeight: 1.8 }}>
{`torch.cuda.memory._record_memory_history(max_entries=100_000)
...
torch.cuda.memory._dump_snapshot("snap.pickle")`}
        </pre>
      </div>
    )
  }

  const yMax = Math.max(tl.peakReserved, tl.peakAllocated, 1)
  const x = (i) => PAD.l + (i / Math.max(series.length - 1, 1)) * (W - PAD.l - PAD.r)
  const y = (v) => PAD.t + (1 - v / yMax) * (H - PAD.t - PAD.b)

  const area = (key) => {
    const top = series.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join('')
    return `${top}L${x(series.length - 1).toFixed(1)},${y(0).toFixed(1)}L${x(0).toFixed(1)},${y(0).toFixed(1)}Z`
  }

  const ticks = yTicks(yMax)
  const hoverPoint = hover != null ? series[hover] : null

  return (
    <>
      <div className="card">
        <h3>Memory over the trace</h3>
        <p className="sub">
          {count(tl.points.length)} events
          {tl.durationUs ? ` spanning ${duration(tl.durationUs)}` : ''}.
          The gap between the two lines is memory held from the driver but not in use.
        </p>

        <svg
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          onMouseLeave={() => setHover(null)}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const px = ((e.clientX - rect.left) / rect.width) * W
            const frac = (px - PAD.l) / (W - PAD.l - PAD.r)
            const i = Math.round(frac * (series.length - 1))
            setHover(i >= 0 && i < series.length ? i : null)
          }}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />
              <text className="axis" x={PAD.l - 8} y={y(t) + 3} textAnchor="end">{bytesTick(t)}</text>
            </g>
          ))}

          <path className="res" d={area('reserved')} />
          <path className="alc" d={area('allocated')} />

          {tl.oomEvents.map((o) => {
            const i = Math.round((o.index / Math.max(tl.points.length - 1, 1)) * (series.length - 1))
            return <line key={o.index} className="oom" x1={x(i)} x2={x(i)} y1={PAD.t} y2={H - PAD.b} />
          })}

          {hoverPoint && (
            <line className="cursor" x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} />
          )}

          <text className="axis" x={PAD.l} y={H - 8}>event 0</text>
          <text className="axis" x={W - PAD.r} y={H - 8} textAnchor="end">
            {count(tl.points.length)}
          </text>
        </svg>

        <div className="legend">
          <span><i style={{ background: 'var(--line-reserved)' }} />reserved — peak {bytes(tl.peakReserved)}</span>
          <span><i style={{ background: 'var(--line-allocated)' }} />allocated — peak {bytes(tl.peakAllocated)}</span>
          {tl.oomEvents.length > 0 && <span><i style={{ background: 'var(--line-oom)' }} />OOM</span>}
          <span className="spacer" style={{ flex: 1 }} />
          {hoverPoint ? (
            <span className="num">
              event {hoverPoint.i} · <code>{hoverPoint.action}</code> · allocated {bytes(hoverPoint.allocated)} · reserved {bytes(hoverPoint.reserved)}
            </span>
          ) : (
            <span className="muted">hover the chart for a value</span>
          )}
        </div>

        {tl.truncated && (
          <p className="note warn">
            The trace is a ring buffer and this one wrapped, so it begins mid-run. Curve shape is
            accurate; the baseline has been shifted up so it never goes negative.
          </p>
        )}
      </div>

      {tl.oomEvents.length > 0 && <OomList events={tl.oomEvents} />}
    </>
  )
}

function OomList({ events }) {
  return (
    <div className="card">
      <h3>Out-of-memory events</h3>
      <p className="sub">
        The allocation that failed, and where it was called from. This is the request the allocator
        could not satisfy — not necessarily the code that wasted the memory.
      </p>
      {events.map((o, i) => {
        const frame = blameFrame(o.frames)
        return (
          <div key={i} className="detail" style={{ marginTop: i ? 12 : 0 }}>
            <div className="row">
              <span>asked for <b>{bytes(o.size)}</b></span>
              {o.deviceFree != null && <><span>·</span><span>{bytes(o.deviceFree)} free on device</span></>}
              <span>·</span><span>event #{o.index}</span>
            </div>
            {frame && <div className="row"><span>at <b className="mono">{frameLabel(frame)}</b></span></div>}
            <Stack frames={displayStack(o.frames)} limit={25} />
          </div>
        )
      })}
    </div>
  )
}

/** Keep the extremes: min and max of each bucket beat averaging them away. */
function downsample(points, targetCols) {
  if (points.length <= targetCols) { return points }
  const step = points.length / targetCols
  const out = []
  for (let c = 0; c < targetCols; c++) {
    const lo = Math.floor(c * step)
    const hi = Math.min(Math.floor((c + 1) * step), points.length)
    let best = points[lo]
    for (let i = lo; i < hi; i++) {
      if (points[i].reserved > best.reserved ||
        (points[i].reserved === best.reserved && points[i].allocated > best.allocated)) {
        best = points[i]
      }
    }
    out.push(best)
  }
  return out
}

function yTicks(max) {
  const step = niceStep(max / 4)
  const out = []
  for (let v = 0; v <= max * 1.0001; v += step) { out.push(v) }
  return out
}

function niceStep(raw) {
  const pow = 2 ** Math.floor(Math.log2(Math.max(raw, 1)))
  for (const m of [1, 1.5, 2, 3, 4, 6, 8]) {
    if (pow * m >= raw) { return pow * m }
  }
  return pow * 8
}

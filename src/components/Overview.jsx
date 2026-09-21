import { bytes, pct, count } from '../lib/format.js'
import { useStore } from '../store.js'

function Tile({ k, v, n, tone }) {
  return (
    <div className={`tile ${tone ?? ''}`}>
      <div className="k">{k}</div>
      <div className="v num">{v}</div>
      {n && <div className="n">{n}</div>}
    </div>
  )
}

export default function Overview({ device, model }) {
  const s = device.stats
  const tl = device.timeline
  const setTab = useStore((st) => st.setTab)

  // Fragmentation only means anything once there is a meaningful amount of
  // free-but-reserved memory. 90% of 4 MiB is not a problem worth a red tile.
  const fragMatters = s.externalFree > 64 * 1024 * 1024
  const fragTone = !fragMatters ? '' : s.fragmentation > 0.8 ? 'bad' : s.fragmentation > 0.5 ? 'warn' : ''
  const utilTone = s.utilisation > 0.8 ? 'good' : s.utilisation < 0.5 ? 'warn' : ''

  return (
    <>
      <div className="tiles">
        <Tile
          k="Reserved"
          v={bytes(s.reserved)}
          n={`${count(s.segmentCount)} segments from cudaMalloc`}
        />
        <Tile
          k="Allocated"
          v={bytes(s.active)}
          n={`${count(s.activeBlocks)} live blocks`}
          tone="good"
        />
        <Tile
          k="Utilisation"
          v={pct(s.utilisation)}
          n="allocated ÷ reserved"
          tone={utilTone}
        />
        <Tile
          k="Free but reserved"
          v={bytes(s.externalFree)}
          n={`${count(s.freeBlocks)} free blocks inside segments`}
          tone={s.externalFree > s.active ? 'warn' : ''}
        />
        <Tile
          k="Fragmentation"
          v={pct(s.fragmentation)}
          n={`largest free block is ${bytes(s.largestFreeBlock)}`}
          tone={fragTone}
        />
        <Tile
          k="Rounding waste"
          v={bytes(s.internalWaste)}
          n={`requested ${bytes(s.requested)} of ${bytes(s.active)}`}
          tone={s.internalWaste > 0.1 * s.active ? 'warn' : ''}
        />
      </div>

      <div className="card">
        <h3>Where the reserved memory went</h3>
        <p className="sub">
          Every byte the allocator holds from the driver, split by what it is doing right now.
        </p>
        <div className="mix">
          <div className="a" style={{ width: `${(s.requested / s.reserved) * 100}%` }} title={`live tensors ${bytes(s.requested)}`} />
          <div className="r" style={{ width: `${(s.internalWaste / s.reserved) * 100}%` }} title={`rounding waste ${bytes(s.internalWaste)}`} />
          <div className="f" style={{ width: `${(s.externalFree / s.reserved) * 100}%` }} title={`free ${bytes(s.externalFree)}`} />
        </div>
        <div className="legend">
          <span><i style={{ background: 'var(--blk-active)' }} />live tensors — {bytes(s.requested)}</span>
          <span><i style={{ background: 'var(--blk-pending)' }} />rounding waste — {bytes(s.internalWaste)}</span>
          <span><i style={{ background: 'var(--blk-free)', border: '1px solid var(--blk-free-edge)' }} />free inside segments — {bytes(s.externalFree)}</span>
        </div>
      </div>

      <Diagnosis device={device} onTab={setTab} />

      {tl.hasTrace && (
        <div className="card">
          <h3>Trace</h3>
          <p className="sub">
            {count(tl.points.length)} recorded events
            {tl.durationUs ? ` over ${(tl.durationUs / 1e6).toFixed(2)}s` : ''}
            {' · '}peak allocated {bytes(tl.peakAllocated)}
            {' · '}peak reserved {bytes(tl.peakReserved)}
          </p>
          <div className="legend">
            {Object.entries(tl.counts).map(([k, n]) => (
              <span key={k}><code>{k}</code> — {count(n)}</span>
            ))}
          </div>
          {tl.truncated && (
            <p className="note warn">
              The trace ran past its <code>max_entries</code> limit, so it starts mid-run and the
              earliest allocations are missing. The curve shape is right; the baseline is shifted so
              it never dips below zero. Pass a larger <code>max_entries</code> to
              <code> _record_memory_history</code> for absolute numbers.
            </p>
          )}
        </div>
      )}

      {model.allocatorSettings && (
        <div className="card">
          <h3>Allocator settings</h3>
          <p className="sub">What the caching allocator was configured with when the snapshot was taken.</p>
          <table>
            <tbody>
              {Object.entries(model.allocatorSettings).map(([k, v]) => (
                <tr key={k}>
                  <td className="mono">{k}</td>
                  <td className="mono muted">{formatSetting(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

function formatSetting(v) {
  if (v === true) { return 'true' }
  if (v === false) { return 'false' }
  if (v === null || v === undefined || v === '') { return '—' }
  return String(v)
}

/**
 * The part that earns the tool its name: read the numbers and say, in words,
 * what they mean. Each finding names the evidence so it can be argued with.
 */
function Diagnosis({ device, onTab }) {
  const s = device.stats
  const tl = device.timeline
  const findings = []

  if (tl.oomEvents.length > 0) {
    const worst = tl.oomEvents[tl.oomEvents.length - 1]
    findings.push({
      tone: 'bad',
      title: `${tl.oomEvents.length} OOM event${tl.oomEvents.length > 1 ? 's' : ''} in the trace`,
      body: `The last one asked for ${bytes(worst.size)} with ${worst.deviceFree != null ? `${bytes(worst.deviceFree)} free on the device` : 'no room left'}. The timeline marks where they fired.`,
      tab: 'timeline',
    })
  }

  if (s.externalFree > 64 * 1024 * 1024 && s.fragmentation > 0.7) {
    findings.push({
      tone: 'warn',
      title: `${bytes(s.externalFree)} is reserved but free, and badly fragmented`,
      body: `The largest single free block is only ${bytes(s.largestFreeBlock)}, so an allocation bigger than that fails even though ${bytes(s.externalFree)} is technically available. This is the classic "OOM with plenty of memory free". Try expandable_segments:True, or a smaller max_split_size_mb.`,
      tab: 'segments',
    })
  } else if (s.externalFree > s.active && s.externalFree > 64 * 1024 * 1024) {
    findings.push({
      tone: 'warn',
      title: `More memory is idle than in use`,
      body: `${bytes(s.externalFree)} sits free inside reserved segments against ${bytes(s.active)} actually live. The largest free block is ${bytes(s.largestFreeBlock)}, so it is reusable — but torch.cuda.empty_cache() would hand it back if another process needs the card.`,
      tab: 'segments',
    })
  }

  if (s.internalWaste > 0.15 * s.active && s.internalWaste > 8 * 1024 * 1024) {
    findings.push({
      tone: 'warn',
      title: `${bytes(s.internalWaste)} lost to allocator rounding`,
      body: `Live blocks total ${bytes(s.active)} but the tensors in them only asked for ${bytes(s.requested)} — ${pct(s.internalWaste / s.active)} overhead. That is size-class rounding; it moves with tensor shapes, not allocator flags.`,
      tab: 'allocations',
    })
  }

  if (s.smallSegments > 0 && s.largeSegments > 0) {
    const smallBytes = device.segments.filter((x) => x.type === 'small').reduce((n, x) => n + x.totalSize, 0)
    if (smallBytes > 32 * 1024 * 1024) {
      findings.push({
        tone: '',
        title: `${count(s.smallSegments)} small-pool segments holding ${bytes(smallBytes)}`,
        body: 'The allocator keeps a separate pool for allocations under 1 MiB. It never lends that memory to large allocations, so it is permanently unavailable to your activations.',
        tab: 'segments',
      })
    }
  }

  if (findings.length === 0) {
    findings.push({
      tone: 'good',
      title: 'Nothing obviously wrong',
      body: `${pct(s.utilisation)} of reserved memory is live, and the largest free block is ${bytes(s.largestFreeBlock)}. If you are here because of an OOM, check the timeline for the peak rather than this end state.`,
      tab: 'timeline',
    })
  }

  return (
    <div className="card">
      <h3>What this looks like</h3>
      <p className="sub">Read off the numbers above. Click a finding to jump to the view that shows it.</p>
      <table>
        <tbody>
          {findings.map((f, i) => (
            <tr key={i} className="click" onClick={() => onTab(f.tab)}>
              <td style={{ width: 4, padding: 0 }}>
                <div style={{
                  width: 3, height: '100%', minHeight: 34, borderRadius: 2,
                  background: f.tone === 'bad' ? 'var(--error)' : f.tone === 'warn' ? 'var(--warn)' : f.tone === 'good' ? 'var(--nv-green)' : 'var(--border-strong)',
                }} />
              </td>
              <td>
                <div style={{ fontWeight: 600, marginBottom: 3 }}>{f.title}</div>
                <div className="muted" style={{ lineHeight: 1.6 }}>{f.body}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

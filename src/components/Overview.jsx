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
  const setPool = useStore((st) => st.setPool)

  // Fragmentation only means anything once there is a meaningful amount of
  // free-but-reserved memory. 90% of 4 MiB is not a problem worth a red tile.
  const fragMatters = s.externalFree > 64 * 1024 * 1024
  const fragTone = !fragMatters ? '' : s.fragmentation > 0.8 ? 'bad' : s.fragmentation > 0.5 ? 'warn' : ''
  const utilTone = s.utilisation > 0.8 ? 'good' : s.utilisation < 0.5 ? 'warn' : ''

  return (
    <>
      {device.pool && <PoolScope pool={device.pool} onAll={() => setPool(null)} />}

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
          n={device.pools.length > 0 && !device.pool
            ? `largest free block is ${bytes(s.largestFreeBlock)}, across pools that cannot lend to each other`
            : `largest free block is ${bytes(s.largestFreeBlock)}`}
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

      {device.pools.length > 0 && <Pools pools={device.pools} current={device.pool} onPick={setPool} />}

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

/** Says, above the numbers, that the numbers are one pool's and not the card's. */
function PoolScope({ pool, onAll }) {
  return (
    <div className={`scope k-${pool.kind}`}>
      <i />
      <div>
        <b>{pool.label} <span className="mono">({pool.poolId.join(', ')})</span></b>
        <span className="muted"> — every number below is this pool alone. </span>
        <button className="linkish" onClick={onAll}>show the whole device</button>
        <div className="muted small">{pool.origin} <span className="faint">Evidence: {pool.evidence}.</span></div>
      </div>
    </div>
  )
}

/**
 * Reserved memory split by the pool that owns it. This is the table that
 * explains a device whose free memory never seems to get used: free bytes in
 * a private pool are free only to that pool's owner.
 */
function Pools({ pools, current, onPick }) {
  const total = pools.reduce((n, p) => n + p.stats.reserved, 0)
  return (
    <div className="card">
      <h3>Memory pools</h3>
      <p className="sub">
        Private pools are walled off. A free block inside one can only serve allocations made into
        that same pool, and <code>torch.cuda.empty_cache()</code> will not release it while the
        pool's owner — the graph, the <code>MemPool</code> — is still alive. Click a pool to look at
        it alone.
      </p>
      <table>
        <thead>
          <tr>
            <th>Pool</th>
            <th className="r">Reserved</th>
            <th className="r">Live</th>
            <th className="r">Free</th>
            <th className="r">Largest free</th>
            <th className="r">Segments</th>
            <th style={{ width: '16%' }} />
          </tr>
        </thead>
        <tbody>
          {pools.map((p) => {
            const st = p.stats
            return (
              <tr key={p.key} className={`click ${current === p ? 'sel' : ''}`} onClick={() => onPick(current === p ? null : p.key)}>
                <td>
                  <div className={`poolname k-${p.kind}`}>
                    <i />{p.label} <span className="mono muted">({p.poolId.join(', ')})</span>
                  </div>
                  <div className="muted small" style={{ marginTop: 3, lineHeight: 1.5 }}>{p.origin}</div>
                </td>
                <td className="r num">{bytes(st.reserved)}</td>
                <td className="r num">{bytes(st.active)}</td>
                <td className="r num muted">{bytes(st.externalFree)}</td>
                <td className="r num muted">{st.freeBlocks ? bytes(st.largestFreeBlock) : '—'}</td>
                <td className="r num muted">{count(st.segmentCount)}</td>
                <td>
                  <div className="mix thin" title={`${pct(st.reserved / total)} of the device; ${pct(st.utilisation)} of it live`}>
                    <div className="a" style={{ width: `${(st.active / total) * 100}%` }} />
                    <div className="f" style={{ width: `${(st.externalFree / total) * 100}%` }} />
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
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

  // Free memory only means something within one pool: a free block in a graph
  // pool cannot serve an ordinary allocation. So from the whole device,
  // fragmentation is judged on the default pool -- where ordinary allocations
  // go -- and the private-pool finding further down speaks for the rest.
  const defaultPool = !device.pool && device.pools.length > 0
    ? device.pools.find((p) => p.kind === 'default') ?? null
    : null
  const fs = device.pool || device.pools.length === 0 ? s : defaultPool?.stats
  const where = defaultPool ? ' in the default pool' : ''
  const inPrivate = device.pool && device.pool.kind !== 'default'
  const fragFix = inPrivate
    ? 'Allocations made into this pool can only use its own free blocks, so the fix is in what goes into it, or in giving it more room.'
    : 'Try expandable_segments:True, or a smaller max_split_size_mb.'

  if (fs && fs.externalFree > 64 * 1024 * 1024 && fs.fragmentation > 0.7) {
    findings.push({
      tone: 'warn',
      title: `${bytes(fs.externalFree)} is reserved but free${where}, and badly fragmented`,
      body: `The largest single free block${where} is only ${bytes(fs.largestFreeBlock)}, so an allocation bigger than that fails even though ${bytes(fs.externalFree)} is technically available. This is the classic "OOM with plenty of memory free". ${fragFix}`,
      tab: 'segments',
    })
  } else if (fs && fs.externalFree > fs.active && fs.externalFree > 64 * 1024 * 1024) {
    findings.push({
      tone: 'warn',
      title: `More memory is idle than in use${where}`,
      body: `${bytes(fs.externalFree)} sits free inside reserved segments${where} against ${bytes(fs.active)} actually live. The largest free block is ${bytes(fs.largestFreeBlock)}, so it is reusable` +
        (inPrivate ? ' — by this pool alone.' : ' — but torch.cuda.empty_cache() would hand it back if another process needs the card.'),
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

  // Only from the whole-device view: inside one pool, "private pools" is the
  // pool you are already looking at.
  const privatePools = device.pool ? [] : device.pools.filter((p) => p.kind !== 'default')
  const privateFree = privatePools.reduce((n, p) => n + p.stats.externalFree, 0)
  const privateReserved = privatePools.reduce((n, p) => n + p.stats.reserved, 0)
  if (privatePools.length > 0 && privateFree > 64 * 1024 * 1024) {
    const graphs = privatePools.filter((p) => p.kind === 'graph')
    const ownPools = graphs.filter((p) => p.poolId[0] > 0 && p.poolId[1] === 0)
    findings.push({
      tone: privateFree > s.active ? 'warn' : '',
      title: `${bytes(privateFree)} free inside private pools, where nothing else can use it`,
      body: `${count(privatePools.length)} private pool${privatePools.length > 1 ? 's hold' : ' holds'} ${bytes(privateReserved)} of the ${bytes(s.reserved)} reserved. Their free blocks cannot serve ordinary allocations, and empty_cache() leaves them alone while the owner is alive.` +
        (graphs.length > 0 ? ' For a CUDA graph that is the cost of replay: its intermediates live at fixed addresses, so the pool stays as big as the capture\'s peak.' : '') +
        (ownPools.length > 0 && graphs.length > 1 ? ` ${count(ownPools.length)} of the graph pools ${ownPools.length > 1 ? 'each belong' : 'belongs'} to a single capture; capturing with pool=torch.cuda.graph_pool_handle() lets graphs that never run at once share one.` : ''),
      tab: 'segments',
    })
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

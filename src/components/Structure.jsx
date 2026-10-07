import { useMemo, useState } from 'react'

import { bytes, addr, pct, count } from '../lib/format.js'
import { BlockState } from '../lib/snapshot.js'
import { buildStructure, layoutStructure, lineage, blockSourceId, COLUMNS } from '../lib/structure.js'
import { useStore } from '../store.js'
import { BlockDetail } from './Segments.jsx'
import Stack from './Stack.jsx'

const W = 1200
const NODE_W = 12
const XS = [0, 210, 420, 640, 880]
const PAD = { t: 26, b: 8 }

/**
 * The allocator as a graph: device -> pool -> free list -> segment -> the code
 * holding it. Column heights are bytes, and each ribbon is split into live
 * (green) and cached (dark), so the place free memory is stuck is the place
 * the dark ribbons pile up.
 *
 * Click any node to follow its bytes through the whole chain; the panel under
 * the graph then says what that node is, and links onward -- a segment to its
 * blocks, a block to the line that allocated it, a line to every segment it
 * sits in.
 */
export default function Structure({ device }) {
  const graph = useMemo(() => {
    const g = buildStructure(device)
    return { ...g, ...layoutStructure(g, { xs: XS, nodeW: NODE_W }) }
  }, [device])

  const [selected, setSelected] = useState(null)
  const [hover, setHover] = useState(null)
  const sel = selected && graph.byId.has(selected) ? selected : null
  const focus = hover ?? sel

  if (device.segments.length === 0) {
    return (
      <div className="card">
        <h3>Allocator structure</h3>
        <p className="sub">Nothing is reserved in this snapshot, so there is no structure to draw.</p>
      </div>
    )
  }

  return (
    <>
      <div className="card">
        <h3>Allocator structure</h3>
        <p className="sub">
          How the caching allocator holds this memory, left to right: the device, the pools it is
          split into, the free lists inside each pool, the segments <code>cudaMalloc</code> returned,
          and the lines of code holding them. Height is bytes. Green ribbons are live tensors, dark
          ones are cached free space — which a request can only reuse from the same free list.
          Click any node to follow it through.
        </p>
        <div className="legend" style={{ marginBottom: 10 }}>
          <span><i style={{ background: 'var(--blk-active)' }} />live</span>
          <span><i style={{ background: 'var(--blk-free-edge)' }} />cached free</span>
          {graph.pools.length > 1 && graph.pools.map((p) => (
            <span key={p.key} className={`k-${p.kind}`}><i style={{ background: 'var(--pool-kind)' }} />{p.label} ({p.poolId.join(', ')})</span>
          ))}
          {sel && (
            <>
              <span className="spacer" style={{ flex: 1 }} />
              <button className="linkish" onClick={() => setSelected(null)}>clear selection</button>
            </>
          )}
        </div>
        <div className="structure-scroll">
          <Graph
            graph={graph}
            focus={focus}
            selected={sel}
            onHover={setHover}
            onSelect={(id) => setSelected(id === sel ? null : id)}
          />
        </div>
      </div>

      {sel
        ? <NodeDetail key={sel} graph={graph} node={graph.byId.get(sel)} onSelect={setSelected} />
        : (
          <div className="card">
            <p className="sub" style={{ margin: 0 }}>
              Select a node above. A <b>segment</b> opens its block map; a <b>block</b> shows the stack
              that allocated it; a <b>source</b> lists every segment it is pinning.
            </p>
          </div>
        )}
    </>
  )
}

function Graph({ graph, focus, selected, onHover, onSelect }) {
  const { nodes, links, byId, height } = graph
  const H = height + PAD.t + PAD.b
  const focusNode = focus ? byId.get(focus) : null

  // What to light up. A source or the free node is a share of every ribbon
  // upstream of it, so it is drawn as an overlay of exactly that share; any
  // other node is a subtree, so its whole lineage lights.
  const lit = useMemo(() => (focus ? lineage(graph, focus) : null), [graph, focus])
  const share = (l) => {
    if (!focusNode) { return null }
    if (focusNode.kind === 'source') { return { live: l.src.get(focus) ?? 0, free: 0 } }
    if (focusNode.kind === 'free') { return { live: 0, free: l.free } }
    return null
  }
  const k = graph.k

  return (
    <svg
      className="chart structure"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Caching allocator structure"
      onMouseLeave={() => onHover(null)}
    >
      {COLUMNS.map((c, i) => (
        <text key={c} className="colhdr" x={XS[i]} y={12}>{c}</text>
      ))}
      <g transform={`translate(0 ${PAD.t})`}>
        <g className="ribbons">
          {links.map((l) => {
            const a = byId.get(l.source)
            const b = byId.get(l.target)
            const x0 = a.x1
            const x1 = b.x0
            const on = !lit || (lit.has(l.source) && lit.has(l.target))
            const sh = share(l)
            const dim = lit && (sh ? true : !on)
            const lw = l.live * k
            const fw = l.free * k
            return (
              <g key={l.id} className={dim ? 'dim' : ''}>
                {lw > 0 && <path className="rb live" d={ribbon(x0, l.sy, x1, l.ty, lw)} />}
                {fw > 0 && <path className="rb free" d={ribbon(x0, l.sy + lw, x1, l.ty + lw, fw)} />}
                {sh && sh.live > 0 && <path className="rb live hl" d={ribbon(x0, l.sy, x1, l.ty, sh.live * k)} />}
                {sh && sh.free > 0 && <path className="rb free hl" d={ribbon(x0, l.sy + lw, x1, l.ty + lw, sh.free * k)} />}
              </g>
            )
          })}
        </g>
        <g className="nodes">
          {nodes.map((n) => {
            const faded = lit && !lit.has(n.id)
            return (
              <g
                key={n.id}
                className={['node', `n-${n.kind}`, n.pool ? `k-${n.pool.kind}` : '', faded ? 'dim' : '', n.id === selected ? 'on' : ''].join(' ')}
                onMouseEnter={() => onHover(n.id)}
                onClick={() => onSelect(n.id)}
              >
                <title>{tooltip(n)}</title>
                {/* A wider invisible target, so a 2px segment is still clickable. */}
                <rect className="hit" x={n.x0 - 3} y={n.y0 - 2} width={NODE_W + 6} height={Math.max(n.y1 - n.y0, 1) + 4} />
                <rect className="box" x={n.x0} y={n.y0} width={NODE_W} height={Math.max(n.y1 - n.y0, 1)} rx={2} />
                {n.kind === 'segment' && n.value > 0 && (
                  <rect className="fill" x={n.x0} y={n.y0} width={NODE_W} height={Math.max(n.y1 - n.y0, 1) * (n.live / n.value)} rx={2} />
                )}
                <NodeLabel n={n} />
              </g>
            )
          })}
        </g>
      </g>
    </svg>
  )
}

function NodeLabel({ n }) {
  const h = n.y1 - n.y0
  const y = (n.y0 + n.y1) / 2
  const x = n.x1 + 6
  if (n.kind === 'segment' || n.kind === 'segrest') {
    // Segments are many; only the tall ones have room to say anything.
    if (h < 13) { return null }
    return (
      <text className="lbl small" x={x} y={y} dy="0.35em">
        {n.kind === 'segment' ? `${bytes(n.value)} · ${pct(n.value ? n.live / n.value : 0)} live` : n.label}
      </text>
    )
  }
  const name = <tspan className="name">{clip(n.label, n.kind === 'source' ? 38 : 30)}</tspan>
  if (n.kind === 'source' || n.kind === 'free') {
    if (n.room < 12) { return null }
    return <text className="lbl" x={x} y={y} dy="0.35em">{name}<tspan className="val"> {bytes(n.value)}</tspan></text>
  }
  // Two lines when there is room, the name alone when there is not, and
  // nothing rather than a pile-up; the tooltip and the panel still have it.
  if (n.room < 13) { return null }
  const y0 = h >= 28 ? y - 4 : n.y0 + 10
  return (
    <text className="lbl" x={x} y={y0}>
      {name}
      {n.room >= 27 && <tspan className="val" x={x} dy="1.25em">{bytes(n.value)}{n.free > 0 ? ` · ${bytes(n.free)} free` : ''}</tspan>}
    </text>
  )
}

function clip(s, n) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s
}

function tooltip(n) {
  const head = n.kind === 'segment' ? `segment ${addr(n.seg.address)}` : n.label
  return `${head}\n${bytes(n.value)} — ${bytes(n.live)} live, ${bytes(n.free)} free`
}

/** A ribbon of width w from (x0, y0) to (x1, y1), as a filled bezier band. */
function ribbon(x0, y0, x1, y1, w) {
  const xm = (x0 + x1) / 2
  return `M${x0},${y0}C${xm},${y0} ${xm},${y1} ${x1},${y1}` +
    `L${x1},${y1 + w}C${xm},${y1 + w} ${xm},${y0 + w} ${x0},${y0 + w}Z`
}

// ---- detail panel -------------------------------------------------------

function NodeDetail({ graph, node, onSelect }) {
  switch (node.kind) {
    case 'segment': return <SegmentDetail graph={graph} node={node} onSelect={onSelect} />
    case 'source': return <SourceDetail graph={graph} node={node} onSelect={onSelect} />
    case 'free': return <FreeDetail graph={graph} node={node} onSelect={onSelect} />
    default: return <GroupDetail graph={graph} node={node} onSelect={onSelect} />
  }
}

function Crumbs({ graph, node, onSelect }) {
  // The chain above a node, as links: device > pool > free list > segment.
  const up = []
  let cur = node
  for (;;) {
    const l = graph.links.find((x) => x.target === cur.id && graph.byId.get(x.source).col === cur.col - 1)
    if (!l) { break }
    cur = graph.byId.get(l.source)
    up.unshift(cur)
  }
  if (!up.length) { return null }
  return (
    <div className="crumbs">
      {up.map((n) => (
        <span key={n.id}>
          <button className="linkish" onClick={() => onSelect(n.id)}>{crumb(n)}</button>
          <span className="faint"> › </span>
        </span>
      ))}
      <span>{crumb(node)}</span>
    </div>
  )
}

function crumb(n) {
  if (n.kind === 'freelist') { return `${n.label} free list` }
  if (n.kind === 'segment') { return `segment ${addr(n.seg.address)}` }
  return n.label
}

function Head({ title, node, children }) {
  return (
    <>
      <h3>{title}</h3>
      <div className="row big">
        <b>{bytes(node.value)}</b>
        <span>{bytes(node.live)} live</span>
        <span>·</span>
        <span>{bytes(node.free)} cached free</span>
        <span>·</span>
        <span>{pct(node.value ? node.live / node.value : 0)} used</span>
      </div>
      {children}
    </>
  )
}

function SegmentDetail({ graph, node, onSelect }) {
  const seg = node.seg
  const nodeIds = useMemo(() => new Set(graph.byId.keys()), [graph])
  const [block, setBlock] = useState(null)
  const [hot, setHot] = useState(null)
  const open = block != null ? seg.blocks[block] : null

  // The sources in this segment, biggest first: the chips under the map.
  const sources = useMemo(() => {
    const m = new Map()
    for (const b of seg.blocks) {
      if (!b.active) { continue }
      const id = blockSourceId(b, nodeIds)
      m.set(id, (m.get(id) ?? 0) + b.size)
    }
    return [...m].sort((a, b) => b[1] - a[1])
  }, [seg, nodeIds])

  const free = seg.blocks.filter((b) => !b.active)
  const largest = Math.max(0, ...free.map((b) => b.size))

  return (
    <div className="card sdetail">
      <Crumbs graph={graph} node={node} onSelect={onSelect} />
      <Head title={`Segment ${addr(seg.address)}`} node={node}>
        <p className="sub">
          One {seg.isExpandable ? 'expandable mapping, grown in place as it fills' : <><code>cudaMalloc</code> of {bytes(seg.totalSize)}</>},
          {' '}{seg.type === 'small' ? 'serving requests of 1 MiB and under' : 'serving requests over 1 MiB'}
          {seg.stream !== 0 && <> on {seg.streamName} <span className="mono faint">(handle {addr(seg.stream)})</span></>}.
          {seg.activeSize > 0
            ? ` It can only go back to the driver once every block in it is free — and ${count(seg.blocks.length - free.length)} still are not.`
            : ' Nothing in it is live, so it is being kept purely as cache; torch.cuda.empty_cache() would release it.'}
          {free.length > 0 && ` Its largest free block is ${bytes(largest)}; a bigger request cannot use this segment.`}
        </p>
      </Head>

      <div className="blocks" style={{ marginTop: 4 }}>
        {seg.blocks.map((b) => {
          const src = b.active ? blockSourceId(b, nodeIds) : null
          return (
            <button
              key={b.index}
              className={[
                b.state === BlockState.ALLOCATED ? 'active' : b.state === BlockState.PENDING_FREE ? 'pending' : 'free',
                block === b.index ? 'on' : '',
                hot && src !== hot ? 'mute' : '',
              ].join(' ')}
              style={{ flexGrow: Math.max(b.size, 1), flexBasis: 0 }}
              title={`${bytes(b.size)} ${b.state}\n${addr(b.address)}`}
              onClick={() => setBlock(block === b.index ? null : b.index)}
            />
          )
        })}
        {seg.unaccounted > 0 && <button className="free" style={{ flexGrow: seg.unaccounted, flexBasis: 0 }} />}
      </div>

      {sources.length > 0 && (
        <div className="chips">
          <span className="faint">held by</span>
          {sources.map(([id, n]) => (
            <button
              key={id}
              className="chip"
              onMouseEnter={() => setHot(id)}
              onMouseLeave={() => setHot(null)}
              onClick={() => onSelect(id)}
              title="Show this source in the graph"
            >
              <span className="mono">{graph.byId.get(id)?.label ?? id}</span> <span className="faint">{bytes(n)}</span>
            </button>
          ))}
        </div>
      )}

      {open && (
        <>
          <BlockDetail seg={seg} block={open} />
          {open.active && (
            <p className="small" style={{ margin: '8px 0 0' }}>
              <button className="linkish" onClick={() => onSelect(blockSourceId(open, nodeIds))}>
                Follow this block&apos;s source through the graph →
              </button>
            </p>
          )}
        </>
      )}
    </div>
  )
}

function SourceDetail({ graph, node, onSelect }) {
  const setTab = useStore((s) => s.setTab)
  const nodeIds = useMemo(() => new Set(graph.byId.keys()), [graph])

  // Every segment this source has bytes in, with how much of it.
  const rows = useMemo(() => {
    return node.segments.map((seg) => {
      let mine = 0, n = 0
      for (const b of seg.blocks) {
        if (b.active && blockSourceId(b, nodeIds) === node.id) { mine += b.size; n++ }
      }
      const segNode = graph.byId.get(`seg:${seg.address}`)
      return { seg, mine, n, target: segNode ? segNode.id : null }
    }).sort((a, b) => b.mine - a.mine)
  }, [node, graph, nodeIds])

  // Segments it shares with someone else, or with free space: the ones this
  // source alone is keeping from being released.
  const pinned = rows.filter((r) => r.mine < r.seg.totalSize)
  const pinnedFree = pinned.reduce((a, r) => a + (r.seg.totalSize - r.seg.activeSize), 0)

  return (
    <div className="card sdetail">
      <Head title={node.other ? 'Other sources' : 'Source'} node={node}>
        {!node.other && <div className="row"><b className="mono">{node.label}</b></div>}
        <p className="sub">
          {count(rows.reduce((a, r) => a + r.n, 0))} live blocks across {count(rows.length)} segment{rows.length === 1 ? '' : 's'}.
          {pinnedFree > 0 && ` Those segments also hold ${bytes(pinnedFree)} of cached free space that cannot go back to the driver while these blocks live.`}
          {node.other && ` Every source past the largest ${count(graph.nodes.filter((n) => n.kind === 'source' && !n.other).length)}, lumped together.`}
        </p>
      </Head>

      {!node.other && (
        <>
          <Stack frames={node.stack} limit={14} />
          <p className="small" style={{ margin: '8px 0 14px' }}>
            <button
              className="linkish"
              onClick={() => { useStore.setState({ expandedBlame: node.id.slice(4) }); setTab('allocations') }}
            >
              Open in Allocations →
            </button>
          </p>
        </>
      )}

      <table>
        <thead>
          <tr>
            <th>Segment</th>
            <th>Free list</th>
            <th className="r">Blocks</th>
            <th className="r">This source</th>
            <th className="r">Of segment</th>
            <th className="r">Segment free</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 60).map((r) => (
            <tr key={r.seg.address} className={r.target ? 'click' : ''} onClick={() => r.target && onSelect(r.target)}>
              <td className="mono">{addr(r.seg.address)} <span className="faint">{bytes(r.seg.totalSize)}</span></td>
              <td>{graph.byId.get(`fl:${r.seg.poolKey}|${r.seg.type}|${r.seg.stream}`)?.label ?? r.seg.type}</td>
              <td className="r num muted">{count(r.n)}</td>
              <td className="r num">{bytes(r.mine)}</td>
              <td className="r num muted">{pct(r.seg.totalSize ? r.mine / r.seg.totalSize : 0)}</td>
              <td className="r num muted">{bytes(r.seg.totalSize - r.seg.activeSize)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 60 && <p className="muted small">… and {count(rows.length - 60)} more segments</p>}
    </div>
  )
}

/** Device, pool, free list, or a lump of small segments: a set of segments. */
function GroupDetail({ graph, node, onSelect }) {
  const segs = node.segments
  const free = segs.flatMap((s) => s.blocks.filter((b) => !b.active))
  const largest = Math.max(0, ...free.map((b) => b.size))
  const children = graph.links
    .filter((l) => l.source === node.id)
    .map((l) => graph.byId.get(l.target))
    .sort((a, b) => b.value - a.value)

  const title = {
    device: `Device ${node.label}`,
    pool: `${node.label} ${node.sub}`,
    freelist: `Free list · ${node.label}`,
    segrest: node.label,
  }[node.kind]

  return (
    <div className={`card sdetail ${node.pool ? `k-${node.pool.kind}` : ''}`}>
      <Crumbs graph={graph} node={node} onSelect={onSelect} />
      <Head title={title} node={node}>
        <p className="sub">
          {node.kind === 'device' && <>Everything the caching allocator has taken from the driver: {count(segs.length)} segments. This is what <code>nvidia-smi</code> bills to the process, less the CUDA context.</>}
          {node.kind === 'pool' && <>{node.pool.origin} <span className="faint">({node.pool.evidence})</span> A free block here can only serve an allocation routed to this pool.</>}
          {node.kind === 'freelist' && (
            <>
              The allocator keeps {node.type} blocks apart from {node.type === 'large' ? 'small' : 'large'} ones, and only
              offers a free block to a request on the stream it came from. So this — {node.pool.label.toLowerCase()},
              {' '}{node.type}, {node.streamName}{node.stream !== 0 && <span className="mono faint"> (handle {addr(node.stream)})</span>} — is
              the set a {node.type === 'small' ? '≤ 1 MiB' : '> 1 MiB'} request on that stream searches for a best fit.
              {' '}{free.length > 0
                ? <>Its largest free block is <b>{bytes(largest)}</b>: anything bigger means a new <code>cudaMalloc</code>, however much is free elsewhere.</>
                : 'It has no free block at all; the next request it gets needs a new segment.'}
            </>
          )}
          {node.kind === 'segrest' && <>Segments too small to draw one by one, lumped so the bytes still add up.</>}
          {' '}{count(free.length)} free block{free.length === 1 ? '' : 's'}.
        </p>
      </Head>
      <table>
        <thead>
          <tr>
            <th>{node.kind === 'freelist' || node.kind === 'segrest' ? 'Segment' : COLUMNS[node.col + 1]}</th>
            <th className="r">Size</th>
            <th className="r">Live</th>
            <th className="r">Free</th>
            <th style={{ width: '30%' }} />
          </tr>
        </thead>
        <tbody>
          {(node.kind === 'segrest' ? segs.map((s) => ({ id: null, seg: s, label: addr(s.address), value: s.totalSize, live: s.activeSize, free: s.totalSize - s.activeSize })) : children)
            .slice(0, 40)
            .map((c) => (
              <tr key={c.id ?? c.seg.address} className={c.id ? 'click' : ''} onClick={() => c.id && onSelect(c.id)}>
                <td className="mono">{c.kind === 'segment' ? addr(c.seg.address) : c.kind === 'freelist' ? `${c.label} free list` : c.label}</td>
                <td className="r num">{bytes(c.value)}</td>
                <td className="r num muted">{bytes(c.live)}</td>
                <td className="r num muted">{bytes(c.free)}</td>
                <td><div className="mix thin"><div className="a" style={{ flex: c.live }} /><div className="f" style={{ flex: c.free }} /></div></td>
              </tr>
            ))}
        </tbody>
      </table>
      {(node.kind === 'segrest' ? segs.length : children.length) > 40 && (
        <p className="muted small">… and {count((node.kind === 'segrest' ? segs.length : children.length) - 40)} more</p>
      )}
    </div>
  )
}

function FreeDetail({ graph, node, onSelect }) {
  // Cached free space, broken down by the free list it is stuck in -- the
  // only grouping that says what it could be reused for.
  const lists = graph.nodes.filter((n) => n.kind === 'freelist' && n.free > 0).map((fl) => {
    const blocks = fl.segments.flatMap((s) => s.blocks.filter((b) => !b.active))
    return { fl, n: blocks.length, largest: Math.max(0, ...blocks.map((b) => b.size)) }
  }).sort((a, b) => b.fl.free - a.fl.free)

  return (
    <div className="card sdetail">
      <h3>Cached free space</h3>
      <div className="row big"><b>{bytes(node.value)}</b><span>reserved, not in use</span></div>
      <p className="sub">
        Memory the allocator keeps instead of returning, so the next allocation skips <code>cudaMalloc</code>.
        It is only as useful as the largest block in the free list a request lands in.
      </p>
      <table>
        <thead>
          <tr>
            <th>Free list</th>
            <th className="r">Free</th>
            <th className="r">Blocks</th>
            <th className="r">Largest</th>
          </tr>
        </thead>
        <tbody>
          {lists.map(({ fl, n, largest }) => (
            <tr key={fl.id} className="click" onClick={() => onSelect(fl.id)}>
              <td>{fl.label}{graph.pools.length > 1 ? <span className="faint"> · {fl.pool.label}</span> : null}</td>
              <td className="r num">{bytes(fl.free)}</td>
              <td className="r num muted">{count(n)}</td>
              <td className="r num muted">{bytes(largest)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

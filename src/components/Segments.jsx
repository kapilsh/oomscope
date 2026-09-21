import { useMemo, useState } from 'react'

import { bytes, addr, pct, count } from '../lib/format.js'
import { blameFrame, frameLabel, displayStack } from '../lib/frames.js'
import { BlockState } from '../lib/snapshot.js'
import { useStore } from '../store.js'
import Stack from './Stack.jsx'

const SORTS = {
  address: (a, b) => a.address - b.address,
  size: (a, b) => b.totalSize - a.totalSize,
  free: (a, b) => (b.totalSize - b.activeSize) - (a.totalSize - a.activeSize),
  fragments: (a, b) => b.blocks.filter((x) => !x.active).length - a.blocks.filter((x) => !x.active).length,
}

/**
 * The memory map: one row per segment, drawn to scale.
 *
 * Widths are proportional within a row rather than across rows, because
 * segments differ by three orders of magnitude and a globally-scaled 2 MiB
 * segment would be a sub-pixel sliver. Each row's own label carries the size,
 * so the row is honest about what it represents.
 */
export default function Segments({ device }) {
  const [sort, setSort] = useState('address')
  const [onlyFragmented, setOnlyFragmented] = useState(false)
  const selected = useStore((s) => s.selectedBlock)
  const selectBlock = useStore((s) => s.selectBlock)

  const segments = useMemo(() => {
    let out = [...device.segments]
    if (onlyFragmented) {
      out = out.filter((s) => s.blocks.some((b) => !b.active) && s.activeSize > 0)
    }
    return out.sort(SORTS[sort])
  }, [device.segments, sort, onlyFragmented])

  const s = device.stats

  return (
    <>
      <div className="card">
        <h3>Segment map</h3>
        <p className="sub">
          Each row is one segment the allocator took from <code>cudaMalloc</code>, drawn to scale
          across its own width. Green is a live tensor, dark is free space the allocator is holding.
          A row that is mostly dark with green specks is a segment that cannot be released and
          cannot be usefully reused. Click any block for its allocation stack.
        </p>
        <div className="legend" style={{ marginBottom: 14 }}>
          <span><i style={{ background: 'var(--blk-active)' }} />allocated</span>
          <span><i style={{ background: 'var(--blk-pending)' }} />pending free</span>
          <span><i style={{ background: 'var(--blk-free)', border: '1px solid var(--blk-free-edge)' }} />free</span>
          <span className="spacer" style={{ flex: 1 }} />
          <label style={{ cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={onlyFragmented}
              onChange={(e) => setOnlyFragmented(e.target.checked)}
              style={{ marginRight: 6, verticalAlign: '-2px' }}
            />
            only partly-used segments
          </label>
          <span>
            sort:{' '}
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              style={{
                background: 'var(--bg-card)', color: 'var(--text)',
                border: '1px solid var(--border-strong)', borderRadius: 5, padding: '2px 6px',
              }}
            >
              <option value="address">address</option>
              <option value="size">size</option>
              <option value="free">free bytes</option>
              <option value="fragments">fragment count</option>
            </select>
          </span>
        </div>

        {segments.length === 0 && <p className="muted">No segments match that filter.</p>}

        {segments.map((seg) => (
          <Segment
            key={`${seg.address}`}
            seg={seg}
            selected={selected}
            onSelect={selectBlock}
          />
        ))}
      </div>

      <div className="card">
        <h3>Free block sizes</h3>
        <p className="sub">
          What the {count(s.freeBlocks)} free blocks look like. An allocation only succeeds if one
          block is big enough on its own — many small blocks and no large one is what fragmentation
          costs you.
        </p>
        <FreeHistogram sizes={s.freeSizes} largest={s.largestFreeBlock} />
      </div>
    </>
  )
}

function Segment({ seg, selected, onSelect }) {
  const free = seg.totalSize - seg.activeSize
  const fragments = seg.blocks.filter((b) => !b.active).length
  const isSel = (b) => selected && selected.segAddress === seg.address && selected.index === b.index
  const openBlock = selected && selected.segAddress === seg.address
    ? seg.blocks.find((b) => b.index === selected.index)
    : null

  return (
    <div className="seg">
      <div className="hdr">
        <span className="mono addr">{addr(seg.address)}</span>
        <b style={{ color: 'var(--text)' }}>{bytes(seg.totalSize)}</b>
        <span className={`tag ${seg.type}`}>{seg.type}</span>
        {seg.isExpandable && <span className="tag exp">expandable</span>}
        {seg.stream !== 0 && <span className="tag">stream {seg.stream}</span>}
        <span>{bytes(seg.activeSize)} live</span>
        <span>·</span>
        <span>{bytes(free)} free{fragments > 1 ? ` in ${fragments} pieces` : ''}</span>
        <span className="spacer" style={{ flex: 1 }} />
        <span className="num">{pct(seg.totalSize ? seg.activeSize / seg.totalSize : 0)} used</span>
      </div>
      <div className="blocks">
        {seg.blocks.map((b) => (
          <button
            key={b.index}
            className={[
              b.state === BlockState.ALLOCATED ? 'active' : b.state === BlockState.PENDING_FREE ? 'pending' : 'free',
              isSel(b) ? 'on' : '',
            ].join(' ')}
            style={{ flexGrow: Math.max(b.size, 1), flexBasis: 0 }}
            title={`${bytes(b.size)} ${b.state}${b.active && b.requestedSize !== b.size ? ` (asked ${bytes(b.requestedSize)})` : ''}\n${addr(b.address)}`}
            onClick={() => onSelect(isSel(b) ? null : { segAddress: seg.address, index: b.index })}
          />
        ))}
        {seg.unaccounted > 0 && (
          <button
            className="free"
            style={{ flexGrow: seg.unaccounted, flexBasis: 0 }}
            title={`${bytes(seg.unaccounted)} not described by the block list`}
          />
        )}
      </div>
      {openBlock && <BlockDetail seg={seg} block={openBlock} />}
    </div>
  )
}

function BlockDetail({ seg, block }) {
  const frame = blameFrame(block.frames)
  const waste = block.size - block.requestedSize
  return (
    <div className="detail">
      <div className="row">
        <span><b>{bytes(block.size)}</b> block</span>
        <span>·</span>
        <span>{block.state.replace('active_', '')}</span>
        <span>·</span>
        <span className="mono">{addr(block.address)}</span>
        <span>·</span>
        <span>offset {bytes(block.offset)} in a {bytes(seg.totalSize)} {seg.type} segment</span>
      </div>
      {block.active && (
        <div className="row">
          <span>tensor asked for <b>{bytes(block.requestedSize)}</b></span>
          {waste > 0 && <><span>·</span><span>{bytes(waste)} lost to rounding ({pct(waste / block.size)})</span></>}
        </div>
      )}
      {frame && (
        <div className="row"><span>allocated at <b className="mono">{frameLabel(frame)}</b></span></div>
      )}
      <Stack frames={displayStack(block.frames)} limit={30} />
    </div>
  )
}

/**
 * Log-scaled buckets: free blocks span bytes to gigabytes, and a linear
 * histogram would put everything in the first bucket.
 */
function FreeHistogram({ sizes, largest }) {
  const buckets = useMemo(() => {
    if (!sizes.length) { return [] }
    const out = []
    for (let e = 10; e <= 31; e++) {
      const lo = 2 ** e
      const hi = 2 ** (e + 1)
      const inBucket = sizes.filter((s) => s >= lo && s < hi)
      if (inBucket.length) {
        out.push({ lo, hi, n: inBucket.length, total: inBucket.reduce((a, b) => a + b, 0) })
      }
    }
    return out
  }, [sizes])

  if (!buckets.length) {
    return <p className="muted">No free blocks — every reserved byte is in use.</p>
  }
  const maxN = Math.max(...buckets.map((b) => b.n))

  return (
    <table>
      <thead>
        <tr>
          <th>Block size</th>
          <th className="r">Count</th>
          <th className="r">Total</th>
          <th style={{ width: '45%' }} />
        </tr>
      </thead>
      <tbody>
        {buckets.map((b) => (
          <tr key={b.lo}>
            <td className="mono">{bytes(b.lo)} – {bytes(b.hi)}</td>
            <td className="r num">{b.n}</td>
            <td className="r num">{bytes(b.total)}</td>
            <td>
              <div
                className="bar"
                style={{
                  width: `${(b.n / maxN) * 100}%`,
                  background: b.hi > largest ? 'var(--nv-green)' : 'var(--blk-free-edge)',
                }}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

import { useMemo, useState } from 'react'

import { bytes, pct, count } from '../lib/format.js'
import { useStore } from '../store.js'
import Stack from './Stack.jsx'

/**
 * Who is holding memory, by the line that allocated it.
 *
 * This is the view that usually ends the investigation: the official viewer
 * shows every allocation as its own box, which is honest but leaves you
 * counting rectangles. Grouping by blame frame turns "3,000 blocks" into
 * "the optimizer, 98 MiB".
 */
export default function Allocations({ device }) {
  const [q, setQ] = useState('')
  const expanded = useStore((s) => s.expandedBlame)
  const toggle = useStore((s) => s.toggleBlame)

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) { return device.blame }
    return device.blame.filter((g) =>
      g.label.toLowerCase().includes(needle) ||
      g.stack.some((f) => `${f.filename} ${f.name}`.toLowerCase().includes(needle)),
    )
  }, [device.blame, q])

  const total = device.stats.active
  const shownBytes = rows.reduce((n, g) => n + g.bytes, 0)
  const max = rows[0]?.bytes ?? 1

  if (device.blame.length === 0) {
    return (
      <div className="card">
        <h3>Live allocations</h3>
        <p className="sub">
          No live blocks in this snapshot, so there is nothing to attribute.
        </p>
      </div>
    )
  }

  return (
    <div className="card">
      <h3>Live allocations by source</h3>
      <p className="sub">
        Every block still alive at snapshot time, grouped by the innermost frame in your own code.
        Rows are the memory you would actually free by changing that line. Click one for the stack.
      </p>

      <div className="legend" style={{ marginBottom: 12 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="filter by file, function, or frame…"
          style={{
            flex: '1 1 260px', background: 'var(--bg-elevated)', color: 'var(--text)',
            border: '1px solid var(--border-strong)', borderRadius: 7, padding: '6px 10px',
          }}
        />
        <span className="num">
          {count(rows.length)} of {count(device.blame.length)} sources · {bytes(shownBytes)}
          {shownBytes !== total && ` of ${bytes(total)}`}
        </span>
      </div>

      <table>
        <thead>
          <tr>
            <th>Source</th>
            <th className="r">Bytes</th>
            <th className="r">Share</th>
            <th className="r">Blocks</th>
            <th className="r">Largest</th>
            <th className="r">Waste</th>
            <th style={{ width: '18%' }} />
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => {
            const open = expanded === g.key
            const waste = g.bytes - g.requested
            return [
              <tr key={g.key} className="click" onClick={() => toggle(g.key)}>
                <td className="mono">
                  <span style={{ color: 'var(--text-faint)', marginRight: 6 }}>{open ? '▾' : '▸'}</span>
                  {g.label}
                </td>
                <td className="r num">{bytes(g.bytes)}</td>
                <td className="r num muted">{pct(total ? g.bytes / total : 0)}</td>
                <td className="r num muted">{count(g.count)}</td>
                <td className="r num muted">{bytes(g.largest)}</td>
                <td className="r num muted">{waste > 0 ? bytes(waste) : '—'}</td>
                <td><div className="bar" style={{ width: `${(g.bytes / max) * 100}%` }} /></td>
              </tr>,
              open && (
                <tr key={`${g.key}-stack`}>
                  <td colSpan={7} style={{ background: 'var(--bg-elevated)' }}>
                    <Stack frames={g.stack} limit={40} />
                  </td>
                </tr>
              ),
            ]
          })}
        </tbody>
      </table>
    </div>
  )
}

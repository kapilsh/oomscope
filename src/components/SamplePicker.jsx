import { useState } from 'react'

import { bytes, pct, count } from '../lib/format.js'
import { samples, groups } from '../samples.js'
import { useStore } from '../store.js'

/**
 * The list of shipped snapshots.
 *
 * Each card leads with what you will see rather than with the filename, and
 * carries the measured headline numbers so the choice can be made without
 * opening three of them. Files are fetched on click, so none of this costs
 * anything until a card is picked.
 */
export default function SamplePicker() {
  const load = useStore((s) => s.load)
  const setTab = useStore((s) => s.setTab)
  const [busy, setBusy] = useState(null)

  const open = async (sample) => {
    setBusy(sample.file)
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}samples/${sample.file}`)
      if (!res.ok) { throw new Error(`${sample.file} returned ${res.status}`) }
      await load({ name: sample.file, buffer: await res.arrayBuffer() })
      // Drop the visitor on the view this sample is actually interesting in,
      // rather than making them find it.
      if (sample.look) { setTab(sample.look) }
    } catch (err) {
      useStore.setState({ error: `Could not load that sample: ${err.message}` })
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="samples">
      {groups.map((g) => {
        const inGroup = samples.filter((s) => s.group === g.id)
        if (inGroup.length === 0) { return null }
        return (
          <section key={g.id}>
            <h3>{g.label}</h3>
            <div className="sample-grid">
              {inGroup.map((s) => (
                <button
                  key={s.file}
                  className="sample"
                  onClick={() => open(s)}
                  disabled={busy !== null}
                >
                  <div className="t">
                    {s.title}
                    {s.synthetic && <span className="tag">synthetic</span>}
                    {busy === s.file && <span className="tag">loading…</span>}
                  </div>
                  <div className="b">{s.blurb}</div>
                  <div className="m num">{facts(s)}</div>
                </button>
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}

/** The two or three numbers that distinguish this sample from the others. */
function facts(s) {
  const out = []
  if (s.reserved > 0) {
    out.push(`${bytes(s.reserved)} reserved`)
    out.push(`${pct(s.utilisation)} used`)
  } else {
    out.push('nothing allocated')
  }
  if (s.devices > 1) { out.push(`${s.devices} devices`) }
  if (s.oomEvents > 0) { out.push(`${s.oomEvents} OOM`) }
  if (!s.hasTrace) {
    out.push('no trace')
  } else if (s.truncated) {
    out.push('trace wrapped')
  } else if (s.traceEvents) {
    out.push(`${count(s.traceEvents)} events`)
  }
  return out.join(' · ')
}

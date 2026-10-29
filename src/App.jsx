import { useCallback, useEffect, useRef, useState } from 'react'

import { bytes } from './lib/format.js'
import { useStore, useDevice, useView, TABS } from './store.js'
import Overview from './components/Overview.jsx'
import Segments from './components/Segments.jsx'
import Allocations from './components/Allocations.jsx'
import Timeline from './components/Timeline.jsx'
import Structure from './components/Structure.jsx'
import SamplePicker from './components/SamplePicker.jsx'

export default function App() {
  const { model, fileName, fileBytes, parseMs, error, loading, tab } = useStore()
  const load = useStore((s) => s.load)
  const clear = useStore((s) => s.clear)
  const setTab = useStore((s) => s.setTab)
  const device = useDevice()
  const view = useView()
  const input = useRef(null)

  // Dropping a file anywhere on the page is the whole interaction, so catch it
  // on the window rather than only on the drop zone -- once a snapshot is open
  // the zone is gone, but dropping another file should still work.
  const [over, setOver] = useState(false)
  useEffect(() => {
    const onOver = (e) => { e.preventDefault(); setOver(true) }
    const onLeave = (e) => { if (e.relatedTarget === null) { setOver(false) } }
    const onDrop = (e) => {
      e.preventDefault()
      setOver(false)
      const f = e.dataTransfer?.files?.[0]
      if (f) { load(f) }
    }
    window.addEventListener('dragover', onOver)
    window.addEventListener('dragleave', onLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragover', onOver)
      window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('drop', onDrop)
    }
  }, [load])

  const pick = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) { load(f) }
    e.target.value = '' // let the same file be re-picked after a clear
  }, [load])

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <Mark />
          oom<span>scope</span>
        </div>
        <div className="tagline">see what your PyTorch memory snapshot is actually holding</div>
        <div className="spacer" />
        {model && (
          <div className="filebar">
            <span className="name mono">{fileName}</span>
            <span>{bytes(fileBytes)} · parsed in {parseMs.toFixed(0)} ms</span>
            <button className="btn" onClick={() => input.current?.click()}>Open another</button>
            <button className="btn" onClick={clear}>Close</button>
          </div>
        )}
      </header>

      <input
        ref={input}
        type="file"
        accept=".pickle,.pkl,application/octet-stream"
        onChange={pick}
        style={{ display: 'none' }}
      />

      {error && (
        <div className="err">
          <b>Could not read that file.</b> {error}
        </div>
      )}

      {loading && <div className="card">Reading snapshot…</div>}

      {!model && !loading && <DropZone over={over} onPick={() => input.current?.click()} />}

      {model && device && (
        <>
          <div className="tabs">
            {TABS.map((t) => (
              <button key={t} className={t === tab ? 'on' : ''} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
            {model.devices.length > 1 && <DevicePicker model={model} />}
          </div>

          {device.pools.length > 0 && <PoolPicker device={device} />}

          {tab === 'overview' && <Overview device={view} model={model} />}
          {tab === 'segments' && <Segments device={view} />}
          {tab === 'structure' && <Structure device={view} />}
          {tab === 'allocations' && <Allocations device={view} />}
          {tab === 'timeline' && <Timeline device={view} />}
        </>
      )}

      <footer>
        <span className="spacer" style={{ flex: 1 }} />
        <a href="https://github.com/kapilsh/oomscope">source</a>
        <a href="https://www.kapilsharma.dev/">kapilsharma.dev</a>
      </footer>
    </div>
  )
}

/**
 * The wordmark's glyph: three segment rows, part live and part stranded -- the
 * segment map at 20px. Drawn inline rather than loaded from public/favicon.svg
 * so it scales with the type around it; the two are the same artwork, so a
 * change to one belongs in the other.
 */
function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <g fill="var(--blk-free)">
        <rect x="6" y="7" width="20" height="4.5" rx="1.6" />
        <rect x="6" y="13.75" width="20" height="4.5" rx="1.6" />
        <rect x="6" y="20.5" width="20" height="4.5" rx="1.6" />
      </g>
      <g fill="var(--nv-green)">
        <rect x="6" y="7" width="13.5" height="4.5" rx="1.6" />
        <rect x="21" y="7" width="5" height="4.5" rx="1.6" />
        <rect x="6" y="13.75" width="4.5" height="4.5" rx="1.6" />
        <rect x="12.5" y="13.75" width="3" height="4.5" rx="1.5" />
        <rect x="17.5" y="13.75" width="2.5" height="4.5" rx="1.25" />
        <rect x="6" y="20.5" width="9.5" height="4.5" rx="1.6" />
      </g>
    </svg>
  )
}

function DevicePicker({ model }) {
  const device = useStore((s) => s.device)
  const setDevice = useStore((s) => s.setDevice)
  return (
    <div className="devpick">
      {model.devices.map((d) => (
        <button key={d.id} className={d.id === device ? 'on' : ''} onClick={() => setDevice(d.id)}>
          cuda:{d.id} · {bytes(d.stats.reserved)}
        </button>
      ))}
    </div>
  )
}

/**
 * Only drawn when the device has a private pool. Every view below follows the
 * choice, so "is this graph pool fragmented" is the same question as "is this
 * device", asked of a smaller set of segments.
 */
function PoolPicker({ device }) {
  const pool = useStore((s) => s.pool)
  const setPool = useStore((s) => s.setPool)
  return (
    <div className="poolpick">
      <span className="lbl">pool</span>
      <button className={pool == null ? 'on' : ''} onClick={() => setPool(null)}>
        all · {bytes(device.stats.reserved)}
      </button>
      {device.pools.map((p) => (
        <button
          key={p.key}
          className={`${p.key === pool ? 'on' : ''} k-${p.kind}`}
          onClick={() => setPool(p.key)}
          title={`${p.origin}\n${p.evidence}`}
        >
          <i />{p.label} <span className="mono">({p.poolId.join(', ')})</span> · {bytes(p.stats.reserved)}
        </button>
      ))}
    </div>
  )
}

function DropZone({ over, onPick }) {
  return (
    <>
      <div className={`drop ${over ? 'over' : ''}`}>
        <h2>Drop a memory snapshot</h2>
        <p>
          The <code>.pickle</code> written by <code>torch.cuda.memory._dump_snapshot()</code>.
          It is read in your browser — nothing is uploaded, and nothing in it is executed.
        </p>
        <button className="btn primary" onClick={onPick}>Choose a file</button>
        <div className="how">
          <h3>Recording one</h3>
          <pre>
<span className="c"># before the part you want to see</span>{'\n'}
torch.cuda.memory.<span className="k">_record_memory_history</span>(max_entries=100_000){'\n'}
{'\n'}
<span className="c"># ... run your model, or catch the OOM ...</span>{'\n'}
{'\n'}
torch.cuda.memory.<span className="k">_dump_snapshot</span>(<span className="k">"snap.pickle"</span>){'\n'}
torch.cuda.memory.<span className="k">_record_memory_history</span>(enabled=None)
          </pre>
        </div>
      </div>

      <div className="or">Or, Open one of these example traces to get started</div>
      <SamplePicker />
    </>
  )
}

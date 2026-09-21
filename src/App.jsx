import { useCallback, useEffect, useRef, useState } from 'react'

import { bytes } from './lib/format.js'
import { useStore, useDevice, TABS } from './store.js'
import Overview from './components/Overview.jsx'
import Segments from './components/Segments.jsx'
import Allocations from './components/Allocations.jsx'
import Timeline from './components/Timeline.jsx'
import SamplePicker from './components/SamplePicker.jsx'

export default function App() {
  const { model, fileName, fileBytes, parseMs, error, loading, tab } = useStore()
  const load = useStore((s) => s.load)
  const clear = useStore((s) => s.clear)
  const setTab = useStore((s) => s.setTab)
  const device = useDevice()
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
        <div className="brand">oom<span>scope</span></div>
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

          {tab === 'overview' && <Overview device={device} model={model} />}
          {tab === 'segments' && <Segments device={device} />}
          {tab === 'allocations' && <Allocations device={device} />}
          {tab === 'timeline' && <Timeline device={device} />}
        </>
      )}

      <footer>
        <span>Everything runs in this tab — the snapshot is never uploaded anywhere.</span>
        <span className="spacer" style={{ flex: 1 }} />
        <a href="https://github.com/kapilsh/oomscope">source</a>
        <a href="https://www.kapilsharma.dev/">kapilsharma.dev</a>
      </footer>
    </div>
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

      <div className="or">or open one of these — real captures, from a real GPU</div>
      <SamplePicker />
    </>
  )
}

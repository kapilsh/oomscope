// Render every view against the demo snapshot and fail on anything thrown.
//
// A build succeeding only proves the modules parse. This proves each view can
// actually render a real snapshot -- the class of bug (a missing field, a bad
// map over an empty array, a divide by a zero-size segment) that otherwise
// shows up as a blank page after deploy.
//
//     node scripts/smoke.mjs                  # public/demo.pickle
//     node scripts/smoke.mjs snapshots/*.pickle
//
// It renders to a string, so there is no DOM and no browser: enough to catch a
// crash, not a substitute for looking at the thing.

import { readFileSync, readdirSync, writeFileSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { build } from 'esbuild'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src')
// Build inside the repo so that react / react-dom, which we deliberately leave
// unbundled, resolve through the repo's own node_modules at import time.
const WORK = join(ROOT, 'node_modules', '.oomscope-smoke')

const ENTRY = `
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement as h } from 'react'

import { unpickle } from '${SRC}/lib/unpickle.js'
import { parseSnapshot } from '${SRC}/lib/snapshot.js'
import Overview from '${SRC}/components/Overview.jsx'
import Segments from '${SRC}/components/Segments.jsx'
import Allocations from '${SRC}/components/Allocations.jsx'
import Timeline from '${SRC}/components/Timeline.jsx'
import SamplePicker from '${SRC}/components/SamplePicker.jsx'
import App from '${SRC}/App.jsx'

export function renderPicker() {
  return renderToStaticMarkup(h(SamplePicker, {})).length
}

// The shell: header and wordmark glyph, drop zone, footer. Effects do not run
// in a static render, which is fine -- this is here to catch a component that
// throws on first paint, like a malformed inline SVG.
export function renderApp() {
  return renderToStaticMarkup(h(App, {})).length
}

export function run(bytes) {
  const model = parseSnapshot(unpickle(bytes))
  const device = model.devices[0]
  const out = {}
  for (const [name, C, props] of [
    ['overview', Overview, { device, model }],
    ['segments', Segments, { device }],
    ['allocations', Allocations, { device }],
    ['timeline', Timeline, { device }],
  ]) {
    out[name] = renderToStaticMarkup(h(C, props)).length
  }
  return { out, model }
}
`

try {
  mkdirSync(WORK, { recursive: true })
  const entryPath = join(WORK, 'entry.jsx')
  const outPath = join(WORK, 'bundle.mjs')
  writeFileSync(entryPath, ENTRY)

  await build({
    entryPoints: [entryPath],
    bundle: true,
    format: 'esm',
    platform: 'node',
    outfile: outPath,
    absWorkingDir: ROOT,
    jsx: 'automatic',
    logLevel: 'error',
    // react-dom/server is CommonJS and calls require() at load time, which a
    // bundled ESM output cannot do. Node loads it fine unbundled.
    external: ['react', 'react-dom', 'react/jsx-runtime', 'react-dom/server'],
  })

  const { run, renderPicker, renderApp } = await import(`file://${outPath}?t=${Date.now()}`)

  const appLen = renderApp()
  if (appLen < 1000) {
    console.error(`app shell rendered only ${appLen} bytes`)
    process.exitCode = 1
  } else {
    console.log(`ok  app shell      ${appLen.toLocaleString()} bytes of markup`)
  }

  // The sample picker is the landing page, so a broken manifest is a blank
  // first impression. Check it before anything else.
  const pickerLen = renderPicker()
  if (pickerLen < 1000) {
    console.error(`sample picker rendered only ${pickerLen} bytes -- manifest broken?`)
    process.exitCode = 1
  } else {
    console.log(`ok  sample picker  ${pickerLen.toLocaleString()} bytes of markup\n`)
  }

  const targets = process.argv.slice(2)
  const files = targets.length
    ? targets
    : readdirSync(join(ROOT, 'public', 'samples'))
      .filter((f) => f.endsWith('.pickle')).sort()
      .map((f) => join(ROOT, 'public', 'samples', f))

  let failures = 0
  for (const file of files) {
    const label = file.split('/').pop()
    let out, model
    try {
      ({ out, model } = run(readFileSync(file)))
    } catch (err) {
      failures++
      console.log(`${label}\n  THREW: ${err.message}\n`)
      continue
    }
    const d = model.devices[0]
    // What counts as "rendered enough" depends on what the snapshot actually
    // contains. A view that correctly draws its empty state is not a failure:
    // a trace-less snapshot SHOULD give a short timeline telling you how to
    // record one. So the floor is per view, and only drops where the data
    // genuinely is not there.
    const hasSegments = !!d && d.segments.length > 0
    const floors = {
      overview: 500, // always has tiles and a diagnosis, even at zero bytes
      segments: hasSegments ? 500 : 1,
      allocations: d && d.blame.length > 0 ? 500 : 1,
      timeline: d && d.timeline.hasTrace ? 500 : 1,
    }
    const marks = Object.entries(out).map(([name, len]) => {
      const ok = len >= floors[name]
      if (!ok) { failures++ }
      return `${ok ? '' : '!'}${name} ${len.toLocaleString()}`
    })
    console.log(
      `ok  ${label.padEnd(34)} ${String(d ? d.segments.length : 0).padStart(3)} seg  ` +
      `${String(d ? d.timeline.points.length : 0).padStart(5)} ev  ` +
      `${marks.join('  ')}`,
    )
  }

  if (failures > 0) {
    console.error(`\n${failures} view(s) rendered suspiciously little or threw`)
    process.exitCode = 1
  } else {
    console.log(`\nall views rendered for ${files.length} snapshot(s)`)
  }
} finally {
  rmSync(WORK, { recursive: true, force: true })
}

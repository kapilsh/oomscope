// The sample snapshots that ship with the app.
//
// Each one is a real capture from a real GPU (the one exception is labelled
// as synthetic), recorded by scripts/make_test_snapshots.py. The prose here is
// hand-written; the numbers come from src/samples.stats.json, which is measured
// off the actual files by scripts/make_samples_manifest.mjs, so a regenerated
// sample cannot leave a stale figure behind on the landing page.
//
// They are fetched on click, never on page load, so having eleven of them costs
// a visitor nothing until they ask for one.

import stats from './samples.stats.json'

/**
 * `group` sorts them into the two things a visitor might be here for: seeing
 * what a real problem looks like, or checking the viewer survives a snapshot
 * with nothing in it.
 */
const SAMPLES = [
  {
    file: '02-training-adam.pickle',
    title: 'Training step',
    blurb: 'A transformer mid-training: parameters, gradients and Adam state all live at once. Start here — the Allocations view splits the three apart by the line that made them.',
    group: 'typical',
    look: 'allocations',
  },
  {
    file: '03-fragmented.pickle',
    title: 'Fragmented',
    blurb: 'Large and small allocations interleaved, then only the large ones freed. Every survivor pins the segment it sits in, so the free space is stranded in pieces too small to reuse.',
    group: 'typical',
    look: 'segments',
  },
  {
    file: '04-oom.pickle',
    title: 'Out of memory',
    blurb: 'Allocating 1 GiB at a time until the card says no, then a second smaller request that also fails. The timeline marks where they fired and the panel shows what was asked for.',
    group: 'typical',
    look: 'timeline',
  },
  {
    file: '07-leak.pickle',
    title: 'Leaking activations',
    blurb: 'Every step appends its output to a list that is never cleared. The timeline is the giveaway: a staircase that climbs and never comes back down.',
    group: 'typical',
    look: 'timeline',
  },
  {
    file: '01-inference-clean.pickle',
    title: 'Clean inference',
    blurb: 'Forward passes under no_grad, nothing retained. What healthy looks like, for contrast — though even here a third of the reserved memory is idle.',
    group: 'typical',
    look: 'overview',
  },
  {
    file: '08-small-pool.pickle',
    title: 'Small-pool churn',
    blurb: 'Two thousand sub-1 MiB tensors with holes punched through them. The allocator keeps these in a separate pool it will never lend to a large allocation.',
    group: 'typical',
    look: 'segments',
  },
  {
    file: '10-expandable-segments.pickle',
    title: 'expandable_segments:True',
    blurb: 'The same training run with expandable segments turned on. Thirty segments collapse to two, and the largest free block goes from 24 MiB to 88 — this is the fix the overview suggests when fragmentation is high.',
    group: 'typical',
    look: 'segments',
  },
  {
    file: '06-truncated-trace.pickle',
    title: 'Truncated trace',
    blurb: 'Recorded with max_entries far too small, so the ring buffer wrapped and the trace begins mid-run. The curve shape is real; the baseline is shifted and flagged rather than quietly clamped.',
    group: 'edge',
    look: 'timeline',
  },
  {
    file: '05-no-trace.pickle',
    title: 'No trace recorded',
    blurb: 'A dump without _record_memory_history. The segment map still works — it is the end state — but there is no history and no allocation stacks to attribute.',
    group: 'edge',
    look: 'timeline',
  },
  {
    file: '09-empty.pickle',
    title: 'Empty',
    blurb: 'A CUDA context with nothing allocated. Zero segments, zero bytes, every ratio undefined.',
    group: 'edge',
    look: 'overview',
  },
  {
    file: '11-multi-gpu-SYNTHETIC.pickle',
    title: 'Two devices',
    blurb: 'Two single-GPU captures stitched into one file so the device picker has something to switch between. The bytes are real; the file is not — no run ever produced it.',
    group: 'edge',
    look: 'overview',
    synthetic: true,
  },
]

/** Samples with their measured numbers attached, dropping any whose file went away. */
export const samples = SAMPLES
  .filter((s) => stats[s.file])
  .map((s) => ({ ...s, ...stats[s.file] }))

export const groups = [
  { id: 'typical', label: 'What a problem looks like' },
  { id: 'edge', label: 'Awkward snapshots' },
]

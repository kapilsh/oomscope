// Turning a 50-frame stack into the one line you actually wanted.
//
// A snapshot frame is {name, filename, line}. A single allocation carries the
// whole unwound stack, and in a real capture most of it is noise: the allocator
// itself, the dispatcher, the autograd codegen, CPython's eval loop. A stack
// from a real run looks like
//
//   [ 0] torch::unwind::unwind()
//   [ 3] c10::cuda::CUDACachingAllocator::...::malloc(...)
//   [11] at::(anonymous namespace)::wrapper_CUDA__empty_strided(...)
//   [31] method_vectorcall_VARARGS_KEYWORDS      <- CPython interpreter guts
//   [33] .../site-packages/torch/nn/modules/module.py:1082 <lambda>
//   [52] /app/train.py:17 <module>               <- the only line you wanted
//
// so the job here is to classify frames and pick the one to blame.

const PY = /\.py$/i
const SITE_PACKAGES = /[/\\](site|dist)-packages[/\\]/
const TORCH_PKG = /[/\\]torch[/\\]/
// CPython's own C sources show up with real filenames, so extension alone does
// not separate "interpreter" from "library".
const CPYTHON_SRC = /[/\\](Python|Objects|Modules|Include)[/\\][^/\\]+\.(c|h)$/
// Code torch.compile generated (inductor's cache, /tmp/torchinductor_<user>/).
// It is Python outside site-packages, but nobody wrote it: blame goes past it
// to the line that called the compiled model.
const GENERATED = /[/\\]torchinductor_[^/\\]*[/\\]/

export const FrameKind = {
  USER: 'user', // your code
  LIBRARY: 'library', // installed Python packages, torch included
  RUNTIME: 'runtime', // C/C++: the allocator, ATen, autograd, CPython
}

/** @returns {'user'|'library'|'runtime'} */
export function frameKind(f) {
  const file = f?.filename ?? ''
  if (!PY.test(file)) { return FrameKind.RUNTIME }
  if (SITE_PACKAGES.test(file) || TORCH_PKG.test(file) || GENERATED.test(file)) { return FrameKind.LIBRARY }
  if (CPYTHON_SRC.test(file)) { return FrameKind.RUNTIME }
  return FrameKind.USER
}

export function isPython(f) { return frameKind(f) !== FrameKind.RUNTIME }

// A C++-only stack is what an allocation made in backward has: it runs on
// autograd's own thread, with no Python under it. Its readable frames are the
// autograd node being applied and the aten op it called; the unwinder's own
// frames come first and say nothing.
const BACKWARD_NODE = /autograd::generated::(\w+)::apply\b/
const ATEN_OP = /\bat::_ops::(\w+)::(?:call|redispatch)\b/
const UNWINDER = /unwind::unwind|CapturedTraceback::gather|gather_with_cpp/

/**
 * The frame to put on the row.
 *
 * Prefer the innermost frame in the user's own code, since that is the line
 * they can actually change. Fall back to the innermost library frame (a torch
 * internal still localises the allocation better than `malloc`), then to
 * anything at all, so a stripped or C++-only stack still gets attributed rather
 * than vanishing into an "unknown" bucket.
 */
export function blameFrame(frames) {
  if (!frames || frames.length === 0) { return null }
  const named = (f) => f?.filename && f.filename !== '??' && f.filename !== ''
  return (
    frames.find((f) => frameKind(f) === FrameKind.USER) ??
    frames.find((f) => frameKind(f) === FrameKind.LIBRARY) ??
    // No Python at all: an allocation from inside the runtime. The autograd
    // node says which backward it was, else the aten op; failing both, a
    // frame that at least names a source file and is not the unwinder.
    frames.find((f) => BACKWARD_NODE.test(f.name ?? '')) ??
    frames.find((f) => ATEN_OP.test(f.name ?? '')) ??
    frames.find((f) => named(f) && !UNWINDER.test(f.name ?? '')) ??
    frames.find((f) => !UNWINDER.test(f.name ?? '')) ??
    frames[0]
  )
}

/** Stable identity for grouping allocations that came from the same line. */
export function frameKey(f) {
  if (!f) { return '<no stack>' }
  return `${f.filename}:${f.line}:${f.name}`
}

/** `train.py:17 <module>` -- the long absolute path is noise in a table. */
export function frameLabel(f) {
  if (!f) { return '<no stack>' }
  if (frameKind(f) === FrameKind.RUNTIME) {
    const bw = BACKWARD_NODE.exec(f.name ?? '')
    if (bw) { return `${bw[1]}  (autograd backward)` }
    const op = ATEN_OP.exec(f.name ?? '')
    if (op) { return `aten::${op[1]}` }
  }
  const file = f.filename ?? '??'
  const base = file.split(/[/\\]/).pop() || file
  const where = f.line ? `${base}:${f.line}` : base
  return f.name ? `${where}  ${f.name}` : where
}

/** Drop the interpreter and allocator plumbing, keep the readable stack. */
export function pythonStack(frames) {
  return (frames ?? []).filter(isPython)
}

/**
 * Collapse a stack for display: Python frames if there are any, otherwise the
 * raw stack, because showing nothing at all is worse than showing C++.
 */
export function displayStack(frames) {
  const py = pythonStack(frames)
  return py.length > 0 ? py : (frames ?? [])
}

/**
 * What a trace event, block or segment says about where it came from beyond
 * its stack: the torch.compile region it was made in (recorded with
 * `_record_memory_history(compile_context=True)`; "N/A" outside one), a tag
 * the allocator put on its own bookkeeping, and -- for an allocation made in
 * backward -- the forward-pass stack where its tensor was created.
 */
export function provenance(x) {
  const ev = x?.allocEvent ?? x
  const cc = ev?.compile_context
  const ff = x?.forwardFrames?.length ? x.forwardFrames : ev?.forward_frames
  return {
    compileContext: typeof cc === 'string' && cc !== 'N/A' ? cc : '',
    internal: typeof ev?.internal_metadata === 'string' ? ev.internal_metadata : '',
    forward: Array.isArray(ff) ? ff.map(String) : [],
  }
}

/** The innermost frame torch.compile mapped to an FX node, if the snapshot has any. */
export function fxFrame(frames) {
  return (frames ?? []).find((f) => f.fx_node_name || f.fx_node_op || f.fx_node_target) ?? null
}

/** `node mm_1 · call_function · aten.mm.default`, or ''. */
export function fxLabel(f) {
  if (!f) { return '' }
  const parts = []
  if (f.fx_node_name) { parts.push(`node ${f.fx_node_name}`) }
  if (f.fx_node_op) { parts.push(f.fx_node_op) }
  if (f.fx_node_target) { parts.push(f.fx_node_target) }
  return parts.join(' · ')
}

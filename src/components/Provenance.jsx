import { fxFrame, fxLabel } from '../lib/frames.js'

/**
 * Where an allocation came from, past its stack: the compiled region, the
 * allocator's own tag, and the FX node of the generated code it ran in. Rows for
 * a `.detail` panel; nothing at all when the snapshot recorded none of it.
 */
export function ProvenanceRows({ p, frames }) {
  const fx = fxLabel(fxFrame(frames))
  return (
    <>
      {p.compileContext && (
        <div className="row"><span>compiled region</span><code className="ctx">{p.compileContext}</code></div>
      )}
      {fx && <div className="row"><span>FX</span><code className="ctx">{fx}</code></div>}
      {p.internal && (
        <div className="row"><span>allocator tag</span><code className="ctx" title="internal_metadata: set by the allocator itself, not by your code">{p.internal}</code></div>
      )}
    </>
  )
}

/**
 * A backward-pass allocation's own stack is autograd's engine; the useful one
 * is the forward stack where the tensor it serves was made, which torch keeps
 * as preformatted traceback text.
 */
export function ForwardStack({ frames }) {
  if (!frames?.length) { return null }
  return (
    <details className="fwd">
      <summary>forward-pass stack — where this tensor was created</summary>
      <pre className="stack mono">{frames.join('').trimEnd()}</pre>
    </details>
  )
}

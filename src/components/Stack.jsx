import { Fragment } from 'react'

import { frameKind, fxLabel } from '../lib/frames.js'

/**
 * A stack, innermost first, colour-coded by whose code it is: your code bright,
 * installed packages dim, runtime/interpreter dimmest. That ordering is what
 * lets you skim past 40 frames of dispatcher to the line you wrote.
 *
 * A frame in code torch.compile generated means little by itself, so when the
 * snapshot was taken with FX traces it carries the graph node it implements and
 * the model source that node came from; both go under the frame.
 */
export default function Stack({ frames, limit = 40 }) {
  if (!frames || frames.length === 0) {
    return <div className="muted mono" style={{ fontSize: 12 }}>no stack recorded for this allocation</div>
  }
  const shown = frames.slice(0, limit)
  return (
    <div className="stack mono">
      {shown.map((f, i) => {
        const kind = frameKind(f)
        const file = f.filename && f.filename !== '??' ? f.filename : null
        const fx = fxLabel(f)
        return (
          <Fragment key={i}>
            <div className={kind} title={`${f.filename}:${f.line}`}>
              {String(i).padStart(2, ' ')}  {file ? `${file}:${f.line}` : '·'}  <span className="f">{f.name}</span>
            </div>
            {fx && <div className="fx">{'    ↳ FX '}{fx}</div>}
            {f.fx_original_trace && (
              <div className="fx orig">{f.fx_original_trace.trimEnd().split('\n').map((l) => `      ${l}`).join('\n')}</div>
            )}
          </Fragment>
        )
      })}
      {frames.length > limit && (
        <div className="runtime">… {frames.length - limit} more frames</div>
      )}
    </div>
  )
}

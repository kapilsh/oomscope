import { frameKind } from '../lib/frames.js'

/**
 * A stack, innermost first, colour-coded by whose code it is: your code bright,
 * installed packages dim, runtime/interpreter dimmest. That ordering is what
 * lets you skim past 40 frames of dispatcher to the line you wrote.
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
        return (
          <div key={i} className={kind} title={`${f.filename}:${f.line}`}>
            {String(i).padStart(2, ' ')}  {file ? `${file}:${f.line}` : '·'}  <span className="f">{f.name}</span>
          </div>
        )
      })}
      {frames.length > limit && (
        <div className="runtime">… {frames.length - limit} more frames</div>
      )}
    </div>
  )
}

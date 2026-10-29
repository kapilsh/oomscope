/**
 * A user_metadata string, shown as it was written. It is whatever the user
 * passed to torch.cuda.memory._set_memory_metadata -- often a JSON object --
 * so it is clipped for a row and kept whole in the tooltip.
 */
export default function Meta({ text, clip = 48, extra }) {
  if (!text) { return null }
  const shown = text.length > clip ? `${text.slice(0, clip - 1)}…` : text
  return (
    <code className="meta" title={`user_metadata: ${text}${extra ? `\n${extra}` : ''}`}>{shown}</code>
  )
}

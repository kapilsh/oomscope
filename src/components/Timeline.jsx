import AllocTimeline from './AllocTimeline.jsx'

/**
 * Memory over the recorded trace: every allocation stacked (AllocTimeline),
 * with reserved drawn over it and the OOMs beside it.
 *
 * The gap between the top of the stack and the reserved line is the whole
 * story of a caching allocator: it grows when a freed block cannot be reused
 * and the allocator takes more from the driver instead. A step up in reserved
 * that never comes back down, while the stack stays flat, is fragmentation
 * happening in front of you.
 */
export default function Timeline({ device }) {
  const tl = device.timeline

  if (!tl.hasTrace && device.pool) {
    return (
      <div className="card">
        <h3>Timeline</h3>
        <p className="sub">
          No recorded event landed in this pool's surviving segments — either the trace was not
          recording when it filled, or the ring buffer has since wrapped past it.
        </p>
      </div>
    )
  }

  if (!tl.hasTrace) {
    return (
      <div className="card">
        <h3>Timeline</h3>
        <p className="sub">
          This snapshot has no trace. The segment view still works — it is the end state — but to
          see memory over time, record history before the run:
        </p>
        <pre className="mono" style={{ margin: 0, color: 'var(--text-dim)', fontSize: 12.5, lineHeight: 1.8 }}>
{`torch.cuda.memory._record_memory_history(max_entries=100_000)
...
torch.cuda.memory._dump_snapshot("snap.pickle")`}
        </pre>
      </div>
    )
  }

  return (
    <AllocTimeline key={device.pool?.key ?? `cuda:${device.id}`} device={device}>
        {device.pool && (
          <p className="note">
            {device.pool.exactTrace
              ? 'Only the events that landed in this pool, by the pool id each trace event carries.'
              : 'Only the events that landed in this pool. This trace does not record pools (torch only started stamping them on events later), so each event is placed by its address inside a segment that still exists at snapshot time, after that segment was allocated.'}
            {' '}The x-axis counts this pool's events, not the device's.
            {device.pool.attributionNote && ` ${device.pool.attributionNote}`}
          </p>
        )}

        {tl.truncated && (
          <p className="note warn">
            The trace is a ring buffer and this one wrapped, so it begins mid-run. What was allocated
            before it began is rebuilt from the frees the trace did see and the blocks still alive at
            the snapshot, and sits at the bottom of the stack; the stack those carry is the one
            recorded on their free.
          </p>
        )}
    </AllocTimeline>
  )
}

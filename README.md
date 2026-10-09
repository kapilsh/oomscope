# oomscope

*See what your PyTorch memory snapshot is actually holding.*

**[www.kapilsharma.dev/oomscope](https://www.kapilsharma.dev/oomscope/)**

Drop in the `.pickle` from `torch.cuda.memory._dump_snapshot()` and get the answer to the
question you opened it with: **why is the allocator holding 40 GiB when my tensors are 12?**

Nothing to hand? The landing page ships sixteen real captures to pick from — a leak, a
fragmented heap, a genuine OOM, the same run with `expandable_segments` on, CUDA graphs,
`MemPool`s and pools named with `user_metadata` — each opening on the view it is interesting in.

Everything runs in the browser. The file is never uploaded, and nothing inside it is executed.

## What it shows

| View | What it answers |
| --- | --- |
| **Overview** | Reserved vs. allocated vs. stranded, and a plain-language read of what those numbers mean — fragmentation, rounding waste, small-pool memory your activations can never use. |
| **Segments** | The memory map: every segment `cudaMalloc` handed over, drawn to scale, block by block. A row that is mostly dark with green specks is a segment that can neither be released nor reused. Click any block for the stack that allocated it. |
| **Structure** | The allocator's own hierarchy as a graph: device → pool → free list (pool × large/small × stream, the set a request can actually reuse from) → segment → the line of code holding it. Heights are bytes, ribbons split live from cached. Click any node to follow its bytes through; a segment opens its blocks, a block its stack, a source every segment it pins. |
| **Allocations** | Every live block grouped by the line in *your* code that made it. This is usually where the investigation ends. |
| **Timeline** | Reserved and allocated over the recorded trace, with OOM events marked. A step up in reserved that never comes back down, while allocated stays flat, is fragmentation happening in front of you. |

## Private pools

CUDA graphs, `torch.compile(mode="reduce-overhead")`, `torch.cuda.MemPool` and symmetric-memory
pools all allocate from *private* pools, which a snapshot tags with `segment_pool_id`. A free
block in one pool cannot serve an allocation from another, so when a snapshot has any, a pool
picker appears under the tabs and every view — numbers, segment map, blame, timeline — can be
narrowed to one pool. The overview adds a per-pool table and calls out memory that is free but
walled off.

The snapshot records a pool's id and nothing else about it, so oomscope says how it decided what
each pool is:

| Pool | How it is recognised |
| --- | --- |
| Default | id `(0, 0)` |
| CUDA graph, own pool | id `(N, 0)` — torch numbers capture-private pools this way |
| CUDA graph, cudagraph trees | id `(0, N)`, allocated from `torch/_inductor/cudagraph_trees.py` |
| CUDA graph, shared handle | id `(0, N)`, every segment on a side stream — *inferred*, and labelled so |
| Symmetric memory | id `(0, N)`, allocated from `torch.distributed._symmetric_memory` |
| MemPool | any other `(0, N)`. NCCL / symmetric pools built as `MemPool(allocator, symmetric=True)` are indistinguishable from this |

How a pool's timeline is split depends on the torch that recorded it. Newer torch (2.16 nightlies
on) stamps a `pool_id` on every trace event, and oomscope uses it as is. Older traces carry no
pool id, so each event is placed by address inside a segment that survives to the snapshot,
after that segment's own `segment_alloc`. That is close to exact for private pools, whose
segments live as long as their owner; events in segments released before the snapshot are
counted against the default pool, and the timeline says which of the two it did.

### Naming pools, and other `user_metadata`

A snapshot never names a pool, but newer torch lets you label what you allocate:

```python
torch.cuda.memory._set_memory_metadata({"pool": "comm-buffers"})
with torch.cuda.use_mem_pool(comm_pool):
    buckets = [torch.empty(...) for _ in range(4)]
torch.cuda.memory._set_memory_metadata("")

torch.cuda.memory._annotate_tensor(buckets[0], "all-reduced first")   # a note, after the fact
```

The string lands as `user_metadata` on trace events only — never on segments or blocks — so
oomscope replays the trace to carry it onto what is alive at the snapshot: every live block shows
the metadata of the `alloc` that made it plus any annotations, every segment the metadata of its
`segment_alloc`. A pool whose events carry a JSON object with a `pool` (or `name`) field is shown
by that name; any other string is shown as written, on pools, segments, blocks, allocation
sources and the timeline cursor, and the allocations filter searches it.

Two things to know when tagging. It needs `_record_memory_history` running, since it only
exists on trace events. And it is per thread: `loss.backward()` allocates on autograd's own
thread, so a `{"phase": "backward"}` set around it labels almost nothing — the sample recorded
for this shows exactly that. It also needs the native caching allocator; a `MemPool` backed by a
pluggable allocator qualifies, a whole-process `change_current_allocator` does not (and that
gives no snapshot at all).

One thing no snapshot can show: in torch 2.8, `torch.distributed._symmetric_memory.empty()` maps
its memory itself, around the caching allocator, so those buffers never appear in a snapshot.

## Recording a snapshot

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... run your model, or catch the OOM ...

torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

`max_entries` caps a ring buffer. If it wraps, the trace starts mid-run and oomscope says so
rather than drawing a curve that begins at a lie.

## Why not the built-in viewer

PyTorch ships one at [pytorch.org/memory_viz](https://pytorch.org/memory_viz), and it is good at
what it does: every allocation as its own rectangle, faithfully. This one is pointed at a
narrower question. Three differences do the work:

- **Stacks are filtered.** A captured stack is ~50 frames, of which ~45 are the allocator, the
  dispatcher, autograd codegen and CPython's eval loop. oomscope classifies each frame as your
  code / an installed package / runtime, and blames the innermost frame you can actually change.
- **Allocations are grouped.** "3,000 live blocks" becomes "the optimizer, 98 MiB, 96 blocks".
- **The numbers are interpreted.** Fragmentation is reported as *the largest free block against
  total free*, because that ratio — not the free total — is what decides whether your next
  allocation fails.

## Development

```bash
npm install
npm run dev      # http://localhost:5173
npm run lint
npm test         # the snapshot reader, against fixtures Python wrote
npm run smoke    # render every view against every shipped sample
npm run samples  # remeasure src/samples.stats.json from public/samples/
npm run build    # -> docs/, which CI deploys
```

### Tests

The unpickler is the one piece where being subtly wrong looks like being right: a misread length
prefix still returns a plausible tree. So it is checked against ground truth rather than by eye.

`scripts/make_fixtures.py` writes each fixture twice — once as Python pickles it, once as
Python's `json` sees it — across protocols 2 through 5, covering every value shape a snapshot
contains and the edges a hand-written decoder gets wrong (integers either side of each opcode's
width boundary, negative longs, memoised shared objects, tuples). `npm test` decodes the pickle
and compares against the JSON.

It was also checked against a real 768 KB snapshot from an actual training run: decoded in JS and
in Python, both serialised to sorted JSON, compared byte for byte. They matched.

`npm run smoke` is the other half — a build only proves the modules parse, so this renders the
landing page and all four views against every shipped sample, failing if any of them throws or
comes back empty. Its floors are per view: a trace-less snapshot *should* render a short
timeline telling you how to record one, and that is not a failure.

### The sample snapshots

`public/samples/` holds sixteen captures, regenerated by `scripts/make_test_snapshots.py`
(needs a GPU). They are chosen for the states a viewer has to survive, not just the
photogenic ones — a snapshot with no trace, one whose trace ring buffer wrapped, and one with
nothing allocated at all are the three that break naive parsers.

Each scenario runs in its own subprocess, which is not fussiness: the caching allocator's state
is global and sticky, so two scenarios in one process contaminate each other's segments, and
`expandable_segments` can only be set before the CUDA context exists.

One exception: `11-multi-gpu-SYNTHETIC.pickle` is stitched from two single-GPU captures by
`scripts/make_multi_gpu_snapshot.py`, because the machine these were recorded on has one GPU and
the device picker needs something to switch between. Its bytes are real; the file is not, and it
is labelled as such in the picker.

`15-symmetric-pool-STANDIN.pickle` is real but has a substitute inside: a symmetric pool is
`MemPool(pg.mem_allocator, symmetric=True)` over NCCL's `ncclMemAlloc`, which torch only hands
out on GPUs with multicast support. The capture uses a two-line `cudaMalloc` allocator loaded
through `CUDAPluggableAllocator` instead — the same way NCCL's is — so the pool, its id, blocks
and snapshot are the genuine code path. The picker tags it "stand-in allocator".

`16-named-pools.pickle` needs a newer torch than the rest — one that writes `pool_id` and
`user_metadata` on trace events (it was recorded on a 2.16 nightly). Run that one scenario from
an environment that has it: `python scripts/make_test_snapshots.py named_pools`.

The headline numbers on each card are measured from the files by
`scripts/make_samples_manifest.mjs` into `src/samples.stats.json`, never typed by hand. CI
regenerates it and fails on a diff, so a regenerated sample cannot leave a stale figure on the
landing page.

`scripts/check_snapshots.mjs` prints what every sample parses to — useful for picking which one
to open when working on a particular view.

The samples add ~9 MB to the repo but nothing to page weight: the built page is ~276 KB and a
sample is fetched only when its card is clicked.

### A note on unpickling

Unpickling arbitrary files is famously unsafe, because the format is a little stack machine with
opcodes that import modules and call them. `src/lib/unpickle.js` implements only the value
opcodes — dicts, lists, tuples, strings, numbers, bools, the memo. `GLOBAL`, `REDUCE`, `BUILD`
and friends are refused by name with an explanatory error. There is no code path that constructs
an object, so a hostile `.pickle` has nothing to reach.

## License

[MIT](LICENSE)

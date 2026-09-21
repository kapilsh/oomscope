# oomscope

*See what your PyTorch memory snapshot is actually holding.*

**[www.kapilsharma.dev/oomscope](https://www.kapilsharma.dev/oomscope/)**

Drop in the `.pickle` from `torch.cuda.memory._dump_snapshot()` and get the answer to the
question you opened it with: **why is the allocator holding 40 GiB when my tensors are 12?**

Everything runs in the browser. The file is never uploaded, and nothing inside it is executed.

## What it shows

| View | What it answers |
| --- | --- |
| **Overview** | Reserved vs. allocated vs. stranded, and a plain-language read of what those numbers mean — fragmentation, rounding waste, small-pool memory your activations can never use. |
| **Segments** | The memory map: every segment `cudaMalloc` handed over, drawn to scale, block by block. A row that is mostly dark with green specks is a segment that can neither be released nor reused. Click any block for the stack that allocated it. |
| **Allocations** | Every live block grouped by the line in *your* code that made it. This is usually where the investigation ends. |
| **Timeline** | Reserved and allocated over the recorded trace, with OOM events marked. A step up in reserved that never comes back down, while allocated stays flat, is fragmentation happening in front of you. |

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
npm run smoke    # render every view against the demo snapshot
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

`npm run smoke` is the other half — a build only proves the modules parse, so this renders all
four views against `public/demo.pickle` and fails if any of them throws or comes back empty.

### The demo snapshot

`public/demo.pickle` is a real capture, not a fixture. `scripts/make_sample_snapshot.py`
(needs a GPU) runs a small transformer for a few steps, deliberately strands free space by
interleaving large and small allocations and freeing only the large ones, then asks for 1 TiB so
the trace contains a genuine OOM.

### A note on unpickling

Unpickling arbitrary files is famously unsafe, because the format is a little stack machine with
opcodes that import modules and call them. `src/lib/unpickle.js` implements only the value
opcodes — dicts, lists, tuples, strings, numbers, bools, the memo. `GLOBAL`, `REDUCE`, `BUILD`
and friends are refused by name with an explanatory error. There is no code path that constructs
an object, so a hostile `.pickle` has nothing to reach.

## License

[MIT](LICENSE)

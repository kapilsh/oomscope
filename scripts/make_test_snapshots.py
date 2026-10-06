#!/usr/bin/env python3
"""Generate a suite of memory snapshots for exercising oomscope.

    python3 scripts/make_test_snapshots.py            # all of them
    python3 scripts/make_test_snapshots.py fragmented # just one

Writes to `public/samples/`, which ships with the app: the landing page lets you
pick one of these instead of hunting for a snapshot of your own. They are
committed, so this only needs running when the set changes.

Each scenario is run in a fresh subprocess. That is not fussiness: the caching
allocator's state is global and sticky, so two scenarios in one process
contaminate each other's segments, and `expandable_segments` can only be set
before the CUDA context exists.

The suite is chosen to cover the states a viewer has to survive, not just the
pretty ones -- an empty snapshot, one with no trace at all, and one whose trace
ring buffer wrapped are the three that break naive parsers.
"""

import argparse
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(__file__).resolve().parent.parent / "public" / "samples"

# name -> (filename, what it is for, env overrides)
SCENARIOS = {}


def scenario(filename, purpose, env=None):
    def wrap(fn):
        SCENARIOS[fn.__name__] = (fn, filename, purpose, env or {})
        return fn
    return wrap


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def _model(d=384, heads=6, layers=4):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
            return x + self.ff(self.ln2(x))

    return nn.Sequential(*[Block() for _ in range(layers)]).cuda(), d


def _train(model, opt, d, steps=3, batch=8, seq=256):
    import torch
    for _ in range(steps):
        x = torch.randn(batch, seq, d, device="cuda")
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

@scenario("01-inference-clean.pickle",
          "High utilisation, few segments. The 'nothing obviously wrong' case.")
def inference_clean():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=50_000)
    model, d = _model()
    model.eval()
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(8, 256, d, device="cuda"))
    return locals()


@scenario("02-training-adam.pickle",
          "Params + grads + Adam state. Blame should name the optimizer step.")
def training_adam():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=4)
    return locals()


@scenario("03-fragmented.pickle",
          "Severe external fragmentation: lots free, no large contiguous block.")
def fragmented():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=2)
    # Interleave large and small, free only the large. Each small survivor pins
    # the segment it sits in, so the freed space is stranded in pieces.
    survivors = []
    for i in range(96):
        big = torch.empty(1024 * 1024 * (2 + i % 8), dtype=torch.uint8, device="cuda")
        small = torch.empty(512 * 1024, dtype=torch.uint8, device="cuda")
        if i % 2 == 0:
            survivors.append(small)
        del big
    return locals()


@scenario("04-oom.pickle",
          "A real OOM under real pressure: OOM panel and timeline marker.")
def oom():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=2)
    # Climb until the card says no, keeping every step alive.
    held, oom_count = [], 0
    try:
        for _ in range(64):
            held.append(torch.empty(1024**3, dtype=torch.uint8, device="cuda"))  # 1 GiB
    except torch.OutOfMemoryError:
        oom_count += 1
    # A second, far smaller failure once the card is full, so the panel has two
    # entries with very different request sizes.
    try:
        torch.empty(4 * 1024**3, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        oom_count += 1
    print(f"    held {len(held)} GiB before OOM, {oom_count} OOM events")
    return locals()


@scenario("05-no-trace.pickle",
          "History never recorded. Timeline must show its empty state, not crash.")
def no_trace():
    # Deliberately no _record_memory_history call.
    import torch
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=2)
    return locals()


@scenario("06-truncated-trace.pickle",
          "max_entries far too small, so the ring buffer wraps mid-run.")
def truncated_trace():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=300)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=4)
    return locals()


@scenario("07-leak.pickle",
          "Activations retained every step: a staircase that never comes down.")
def leak():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    leaked = []
    for _ in range(8):
        x = torch.randn(8, 256, d, device="cuda")
        out = model(x)
        leaked.append(out.detach())  # the bug
        out.square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return locals()


@scenario("08-small-pool.pickle",
          "Thousands of sub-1MiB tensors: the small-pool segments finding.")
def small_pool():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    kept = [torch.empty(200 * 1024, dtype=torch.uint8, device="cuda") for _ in range(2000)]
    del kept[::3]  # punch holes through the small pool
    return locals()


@scenario("09-empty.pickle",
          "Nothing allocated at all. Guards every divide-by-reserved.")
def empty():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=10_000)
    # Touch CUDA so a context exists, then give everything back.
    t = torch.empty(1024, device="cuda")
    del t
    torch.cuda.empty_cache()
    return locals()


@scenario("10-expandable-segments.pickle",
          "expandable_segments:True -- segments carry the expandable flag.",
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def expandable_segments():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train(model, opt, d, steps=3)
    return locals()


def _warmup(fn, *args):
    # Graph capture must not be the first time a kernel runs: cuBLAS handles,
    # workspaces and lazy init all allocate, and none of that may happen inside
    # a capture. Warm up on a side stream, as the docs prescribe.
    import torch
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)


@scenario("12-cuda-graphs.pickle",
          "Manual CUDA graph capture: one shared pool across batch sizes, one graph on its own.")
def cuda_graphs():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    model.eval()
    seq = 256

    # The vLLM pattern: capture one graph per batch size, largest first, all
    # into one pool from graph_pool_handle(). Each smaller graph reuses the
    # blocks the larger one left behind, so the pool is sized by the biggest.
    shared = torch.cuda.graph_pool_handle()
    graphs, static_in, static_out = {}, {}, {}
    with torch.no_grad():
        for bs in (16, 8, 4, 2, 1):
            static_in[bs] = torch.randn(bs, seq, d, device="cuda")
            _warmup(model, static_in[bs])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=shared):
                static_out[bs] = model(static_in[bs])
            graphs[bs] = g

        # A second model captured without pool=: it gets a private pool of its
        # own that nothing else can borrow from, sized by its own peak.
        head, _ = _model(layers=2)
        head.eval()
        head_in = torch.randn(16, seq, d, device="cuda")
        _warmup(head, head_in)
        head_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(head_graph):
            head_out = head(head_in)

        for _ in range(3):
            for bs, g in graphs.items():
                static_in[bs].normal_()
                g.replay()
            head_graph.replay()
        torch.cuda.synchronize()
    return locals()


@scenario("13-cudagraph-trees.pickle",
          "torch.compile(mode='reduce-overhead'): cudagraph trees own a shared private pool.")
def cudagraph_trees():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    compiled = torch.compile(model, mode="reduce-overhead")
    for _ in range(4):
        x = torch.randn(8, 256, d, device="cuda")
        loss = compiled(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return locals()


@scenario("14-mempool.pickle",
          "torch.cuda.MemPool: a KV cache fenced off, and a scratch pool that kept its memory.")
def mempool():
    import torch
    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    model.eval()

    # A KV cache carved out up front, in a pool of its own so the activations
    # around it can never fragment the space it needs.
    kv_pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(kv_pool):
        kv = [torch.empty(2, 16, 1024, d, dtype=torch.float16, device="cuda") for _ in range(4)]

    # A scratch pool used for one burst of work and then let go of. Freeing the
    # tensors returns the blocks to the pool, not to the default allocator, so
    # the memory stays reserved and unusable by anything outside it.
    scratch_pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(scratch_pool):
        scratch = [torch.empty(1024 * 1024 * (4 + i % 5), dtype=torch.uint8, device="cuda")
                   for i in range(24)]
    del scratch[1::2]

    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(8, 256, d, device="cuda"))
    return locals()


_STANDIN_ALLOCATOR = r"""
#include <stddef.h>
typedef struct CUstream_st* cudaStream_t;
extern int cudaMalloc(void**, size_t);
extern int cudaFree(void*);
void* standin_alloc(size_t size, int device, cudaStream_t stream) {
  void* p = 0; cudaMalloc(&p, size); return p;
}
void standin_free(void* p, size_t size, int device, cudaStream_t stream) { cudaFree(p); }
"""


@scenario("15-symmetric-pool-STANDIN.pickle",
          "MemPool(symmetric=True) over a pluggable allocator, standing in for ncclMemAlloc.")
def symmetric_pool_standin():
    # The real thing is MemPool(pg.mem_allocator, symmetric=True), with NCCL's
    # ncclMemAlloc behind it. torch 2.8 only hands out that allocator on a GPU
    # with multicast support, and the card these were recorded on has none. So
    # the allocator here is a two-line cudaMalloc wrapper loaded the same way
    # NCCL's is -- through CUDAPluggableAllocator -- and everything above it
    # (the pool, its id, its blocks, the snapshot) is the real code path.
    #
    # Note that torch.distributed._symmetric_memory.empty() would not have
    # worked as a stand-in either: in 2.8 it maps memory itself, around the
    # caching allocator, so its buffers never appear in a snapshot at all.
    import subprocess
    import tempfile
    import torch
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "standin.c").write_text(_STANDIN_ALLOCATOR)
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", tmp / "standin.so", tmp / "standin.c"], check=True)
    alloc = torch.cuda.memory.CUDAPluggableAllocator(str(tmp / "standin.so"), "standin_alloc", "standin_free")

    torch.cuda.memory._record_memory_history(max_entries=80_000)
    model, d = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    symm_pool = torch.cuda.MemPool(alloc.allocator(), symmetric=True)
    with torch.cuda.use_mem_pool(symm_pool):
        # Communication buffers: one per bucket of gradients, plus a workspace
        # that is resized once -- the old one goes back to the pool, not the GPU.
        buckets = [torch.empty(25 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda") for _ in range(4)]
        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        del workspace
        workspace = torch.empty(48 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    _train(model, opt, d, steps=2)
    return locals()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_one(name):
    import torch
    fn, filename, _purpose, _env = SCENARIOS[name]
    # Hold on to what the scenario built. Each one returns its locals(), and if
    # that dict is dropped the tensors are freed before the dump -- which
    # silently produces a snapshot of an idle process rather than of the state
    # the scenario set up.
    state = fn()
    assert state is not None, f"{name} must return locals() so its tensors stay alive"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    torch.cuda.memory._dump_snapshot(str(path))
    torch.cuda.memory._record_memory_history(enabled=None)
    size = path.stat().st_size
    print(f"    {filename}  {size / 1024:.0f} KiB  "
          f"allocated {torch.cuda.memory_allocated() / 2**20:.1f} MiB  "
          f"reserved {torch.cuda.memory_reserved() / 2**20:.1f} MiB")
    # Handed back rather than dropped here: the caller exits without teardown,
    # and freeing a locals() dict destroys its contents in no particular order
    # -- a MemPool can go before the tensors that live in it, which segfaults.
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", help="run a single scenario by name")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        state = run_one(args.child)  # noqa: F841 -- held until exit
        # Skip interpreter teardown. The child exists only to write one file,
        # and tearing down a scenario that owns MemPools can segfault on the
        # way out -- after the snapshot is safely on disk, but failing the run
        # all the same.
        sys.stdout.flush()
        os._exit(0)

    names = [args.only] if args.only else list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; have: {', '.join(SCENARIOS)}")
        _fn, filename, purpose, env = SCENARIOS[name]
        print(f"\n{name}")
        print(f"  {purpose}")
        proc_env = {**os.environ, **env}
        if env:
            print(f"  env: {' '.join(f'{k}={v}' for k, v in env.items())}")
        r = subprocess.run(
            [sys.executable, __file__, "--child", name],
            env=proc_env, capture_output=True, text=True,
        )
        # torch's unwinder warns on some frames; it is noise, not failure.
        for line in r.stdout.splitlines():
            if "unwind" not in line:
                print(line)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"scenario {name} failed")

    print(f"\nwrote {len(names)} snapshot(s) to {OUT}")


if __name__ == "__main__":
    main()

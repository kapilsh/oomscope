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
    del state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", help="run a single scenario by name")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_one(args.child)
        return

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

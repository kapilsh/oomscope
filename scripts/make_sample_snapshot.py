#!/usr/bin/env python3
"""Record the demo snapshot that ships with the app.

Needs a CUDA device. Writes `public/demo.pickle`, which Vite copies into the
build, so the page has something to show before you have a snapshot of your own.

    python3 scripts/make_sample_snapshot.py

The run is contrived on purpose. It has to contain, in one file, every shape the
views claim to handle:

  * a real training step, so parameters / gradients / optimizer state all show up
    with honest stack traces
  * both allocator pools: tensors over 1 MiB land in `large` segments, the small
    ones in `small`
  * deliberate fragmentation -- interleaved allocations of mixed sizes, with the
    large ones freed and the small ones kept, which strands free space in pieces
  * a caught OOM, so the timeline has an `oom` marker and the OOM panel has
    something to explain
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


def train_steps(model, opt, steps, batch, seq, d):
    for _ in range(steps):
        x = torch.randn(batch, seq, d, device="cuda")
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def strand_free_space():
    """Leave the allocator holding memory it cannot hand back.

    Freeing the big tensors returns their blocks to the pool, but the small
    survivors sit in the middle of those segments, so the free space is split
    into pieces and no segment is empty enough to release.
    """
    survivors = []
    for i in range(32):
        big = torch.empty(1024 * 1024 * (2 + i % 6), dtype=torch.uint8, device="cuda")
        small = torch.empty(768 * 1024, dtype=torch.uint8, device="cuda")
        if i % 2 == 0:
            survivors.append(small)
        del big
    return survivors


def provoke_oom():
    """Ask for more than any card has, and let the allocator record the failure."""
    try:
        torch.empty(1024**4, dtype=torch.uint8, device="cuda")  # 1 TiB
    except torch.OutOfMemoryError:
        return True
    return False


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this script needs a CUDA device")

    torch.cuda.memory._record_memory_history(max_entries=60_000)

    d, heads, layers = 384, 6, 4
    model = nn.Sequential(*[Block(d, heads) for _ in range(layers)]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_steps(model, opt, steps=3, batch=8, seq=256, d=d)
    survivors = strand_free_space()
    oomed = provoke_oom()

    out = "public/demo.pickle"
    torch.cuda.memory._dump_snapshot(out)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"wrote {out}")
    print(f"  allocated {torch.cuda.memory_allocated() / 2**20:8.1f} MiB")
    print(f"  reserved  {torch.cuda.memory_reserved() / 2**20:8.1f} MiB")
    print(f"  oom recorded: {oomed}")
    print(f"  survivors kept alive: {len(survivors)}")


if __name__ == "__main__":
    main()

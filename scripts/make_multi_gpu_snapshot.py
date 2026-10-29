#!/usr/bin/env python3
"""Stitch two single-GPU snapshots into one multi-device snapshot.

    python3 scripts/make_multi_gpu_snapshot.py

This one is SYNTHETIC, unlike everything else in snapshots/. It exists because
the device picker only appears when a snapshot has more than one device, and
the machine these were recorded on has a single GPU. The bytes in it are real
-- they came from two real captures -- but no run ever produced this file.

Do not use it to reason about multi-GPU behaviour. Use it to check that the
picker renders, switches, and keeps per-device numbers separate.
"""

import pathlib
import pickle

OUT = pathlib.Path(__file__).resolve().parent.parent / "public" / "samples"
SOURCES = ["02-training-adam.pickle", "03-fragmented.pickle"]
TARGET = "11-multi-gpu-SYNTHETIC.pickle"


def main():
    loaded = []
    for name in SOURCES:
        path = OUT / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run make_test_snapshots.py first")
        loaded.append(pickle.loads(path.read_bytes()))

    merged = {
        "segments": [],
        "device_traces": [],
        "allocator_settings": loaded[0].get("allocator_settings"),
        "external_annotations": loaded[0].get("external_annotations", []),
    }

    for device, snap in enumerate(loaded):
        for seg in snap["segments"]:
            seg = dict(seg)
            seg["device"] = device
            merged["segments"].append(seg)
        # device_traces is indexed by device, so each source's trace has to land
        # at its new index. A source may carry several (mostly empty) lists; the
        # one with events is the device that was actually used.
        traces = snap.get("device_traces") or [[]]
        busiest = max(traces, key=len) if traces else []
        merged["device_traces"].append(busiest)

    path = OUT / TARGET
    path.write_bytes(pickle.dumps(merged, protocol=4))

    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KiB)")
    for device, snap in enumerate(loaded):
        segs = [s for s in merged["segments"] if s["device"] == device]
        total = sum(s["total_size"] for s in segs)
        print(f"  cuda:{device}  from {SOURCES[device]}  "
              f"{len(segs)} segments, {total / 2**20:.1f} MiB reserved, "
              f"{len(merged['device_traces'][device])} trace events")


if __name__ == "__main__":
    main()

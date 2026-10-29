#!/usr/bin/env python3
"""Generate test fixtures for the JS unpickler.

Each fixture is a pair: `<name>.pickle` as Python writes it, and `<name>.json`
as the ground truth the JS decoder has to reproduce. Run this after changing
anything about which shapes we claim to support:

    python3 scripts/make_fixtures.py

Requires only the standard library. `make_sample_snapshot.py` is the one that
needs a GPU; this one does not.
"""

import json
import pathlib
import pickle

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Every value shape a memory snapshot can contain, plus the edges that a
# hand-written decoder gets wrong: ints either side of each opcode's width
# boundary, negative longs, a shared object that must come back through the memo
# rather than being decoded twice, and a tuple (segment_pool_id is one).
shared = {"shared": True}
TYPES = {
    "none": None,
    "true": True,
    "false": False,
    "empty_dict": {},
    "empty_list": [],
    "empty_tuple": (),
    "int_0": 0,
    "int_255": 255,  # BININT1 boundary
    "int_256": 256,  # -> BININT2
    "int_65535": 65535,
    "int_65536": 65536,  # -> BININT
    "int_2p31": 2**31,  # -> LONG1
    "int_2p48": 2**48,  # realistic device address magnitude
    "addr": 138772641480704,  # an address from a real snapshot
    "time_us": 1789995997101492,  # a time_us from a real snapshot
    "neg_small": -1,
    "neg_big": -(2**40),
    "float": 3.5,
    "float_neg": -0.125,
    "ascii": "alloc",
    "unicode": "segment → block µs \U0001f9ea",
    "long_string": "x" * 300,  # -> BINUNICODE rather than SHORT_BINUNICODE
    "tuple2": (0, 0),
    "nested": {"a": [1, {"b": (2, 3)}], "c": [[], {}]},
    "shared_a": shared,
    "shared_b": shared,  # same object: exercises BINGET/LONG_BINGET
    "list_of_dicts": [{"i": i, "s": f"f{i}"} for i in range(40)],
}

# The shape of a snapshot, minus the bulk. Keeps the parser honest about the
# actual key names without committing a megabyte of frames to the repo.
SNAPSHOT_SHAPE = {
    "segments": [
        {
            "device": 0,
            "address": 138772641480704,
            "total_size": 33554432,
            "allocated_size": 2097152,
            "active_size": 2097152,
            "requested_size": 2000000,
            "stream": 0,
            "segment_type": "large",
            "segment_pool_id": (0, 0),
            "is_expandable": False,
            "frames": [],
            "blocks": [
                {
                    "address": 138772641480704,
                    "size": 2097152,
                    "requested_size": 2000000,
                    "state": "active_allocated",
                    "frames": [
                        {"name": "forward", "filename": "/app/model.py", "line": 42},
                        {"name": "<module>", "filename": "/app/train.py", "line": 7},
                    ],
                },
                {
                    "address": 138772643577856,
                    "size": 31457280,
                    "requested_size": 0,
                    "state": "inactive",
                    "frames": [],
                },
            ],
        }
    ],
    "device_traces": [
        [
            {
                "action": "segment_alloc",
                "addr": 138772641480704,
                "size": 33554432,
                "stream": 0,
                "time_us": 1789995997101492,
                "compile_context": "N/A",
                "frames": [],
            },
            {
                "action": "alloc",
                "addr": 138772641480704,
                "size": 2097152,
                "stream": 0,
                "time_us": 1789995997101592,
                "compile_context": "N/A",
                "frames": [{"name": "forward", "filename": "/app/model.py", "line": 42}],
            },
            {
                "action": "free_completed",
                "addr": 138772641480704,
                "size": 2097152,
                "stream": 0,
                "time_us": 1789995997101692,
                "compile_context": "N/A",
                "frames": [],
            },
            {
                "action": "oom",
                "size": 1073741824,
                "device_free": 12345678,
                "stream": 0,
                "time_us": 1789995997101792,
                "frames": [],
            },
        ]
    ],
    "allocator_settings": {
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "max_split_size": -1,
        "expandable_segments": False,
    },
    "external_annotations": [],
}


def emit(name, obj, protocol):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.pickle").write_bytes(pickle.dumps(obj, protocol=protocol))
    # sort_keys so the JS side can compare without caring about insertion order.
    (OUT / f"{name}.json").write_text(json.dumps(obj, sort_keys=True, indent=1))
    print(f"  {name}.pickle (protocol {protocol})")


if __name__ == "__main__":
    print(f"writing fixtures to {OUT}")
    # torch writes protocol 4 today, but pin nothing: decode every protocol the
    # reader claims to accept, so a torch bump to 5 cannot surprise us.
    for proto in (2, 3, 4, 5):
        emit(f"types_p{proto}", TYPES, proto)
    emit("snapshot_shape", SNAPSHOT_SHAPE, 4)
    print("done")

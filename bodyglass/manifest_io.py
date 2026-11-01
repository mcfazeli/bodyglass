"""Safe manifest writes: Phase 2 (structures), scan alignment and veins run at the same time and all
add keys to manifest.json. Every write happens under a lock and replaces the file atomically.

shortcut: the lock is per process; tools that edit a case from another process must not run while the
app is processing that case. Switch to fcntl.flock if a second writer process is ever added."""
import json
import os
import threading

_LOCK = threading.Lock()


def _dump(case_dir, mf):
    path = os.path.join(case_dir, "manifest.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(mf, fh, indent=1)
    os.replace(path + ".tmp", path)   # a crash mid-write never leaves a truncated manifest


def write(case_dir, mf: dict):
    with _LOCK:
        _dump(case_dir, mf)


def update(case_dir, changes: dict):
    """Merge `changes` (only the keys the caller owns) into the manifest on disk."""
    with _LOCK:
        with open(os.path.join(case_dir, "manifest.json")) as fh:
            cur = json.load(fh)
        cur.update(changes)
        _dump(case_dir, cur)
        return cur

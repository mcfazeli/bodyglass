"""Concurrent background writers must not lose each other's keys, and writes are atomic."""
import json
import os
import threading

from bodyglass import manifest_io


def test_concurrent_updates_keep_every_key(tmp_path):
    manifest_io.write(str(tmp_path), {"caseId": "x"})

    def worker(n):
        for i in range(25):
            manifest_io.update(str(tmp_path), {f"k{n}": i})
    ts = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    mf = json.load(open(tmp_path / "manifest.json"))
    assert mf == {"caseId": "x", **{f"k{n}": 24 for n in range(8)}}
    assert not os.path.exists(tmp_path / "manifest.json.tmp")

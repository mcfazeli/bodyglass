"""AI model downloads: host allowlist, checksums, resume/pause, disk check, convert-then-delete (all offline)."""
import hashlib
import io
import os
import time
from collections import namedtuple

import pytest

from bodyglass import ai_models as am


def test_only_https_to_hugging_face():
    assert am.host_ok("https://huggingface.co/qvac/MedPsy-4B/resolve/x/model.safetensors")
    assert am.host_ok("https://cas-bridge.xethub.hf.co/x") and am.host_ok("https://cdn-lfs-us-1.hf.co/x")
    for bad in ("http://huggingface.co/x", "https://evilhuggingface.co/x", "https://huggingface.co.evil.com/x",
                "https://example.com/x", "file:///etc/passwd"):
        assert not am.host_ok(bad)
    with pytest.raises(OSError):
        am.fetch("https://example.com/x", "/tmp/never", 1)


def _f(path, data, lfs=True):
    return {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest() if lfs else None,
            "blob": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}


class FakeOpener:
    """Serves bytes honouring Range, like the Hub/CDN."""
    def __init__(self, blobs):
        self.blobs, self.ranges = blobs, []

    def open(self, req, timeout=None):
        data = self.blobs[req.full_url.rsplit("/", 1)[1]]
        rng = req.headers.get("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        r = io.BytesIO(data[start:])
        r.status = 206 if rng else 200
        r.__enter__ = lambda: r
        return Resp(r)


class Resp:
    def __init__(self, r): self.r, self.status = r, r.status
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self, n): return self.r.read(n)


def test_fetch_pauses_and_resumes_with_range(tmp_path, monkeypatch):
    data = os.urandom(5 * am.CHUNK + 7)
    op = FakeOpener({"w.bin": data})
    dest = str(tmp_path / "w.bin")
    calls = []
    assert am.fetch("https://huggingface.co/r/resolve/x/w.bin", dest, len(data), stop=lambda: len(calls) >= 2,
                    progress=calls.append, opener=op) is False
    assert os.path.getsize(dest + ".part") == 2 * am.CHUNK and not os.path.exists(dest)
    assert am.fetch("https://huggingface.co/r/resolve/x/w.bin", dest, len(data), opener=op)
    assert op.ranges == [None, f"bytes={2 * am.CHUNK}-"]
    assert am.file_ok(dest, _f("w.bin", data))
    small = tmp_path / "config.json"
    small.write_bytes(b"{}")
    assert am.file_ok(str(small), _f("config.json", b"{}", lfs=False))
    assert not am.file_ok(str(small), _f("config.json", b"[]", lfs=False))


def _manifest(files):
    return {"models": [{"id": "m", "name": "M", "repo": "o/M", "revision": "abc", "runtime": "mlx_lm", "bits": [4, 8],
                        "diskGB": {"4": 0.001, "8": 0.002}, "ramGB": {"4": 1}, "files": files}]}


def _wait(mgr, mid="m"):
    for _ in range(200):
        if mgr.state[mid]["status"] not in ("downloading", "converting"):
            return mgr.state[mid]
        time.sleep(0.02)
    raise AssertionError(mgr.state[mid])


def test_download_verify_convert_then_delete_originals(tmp_path, monkeypatch):
    blobs = {"model.safetensors": os.urandom(3000), "config.json": b'{"a": 1}'}
    files = [_f("model.safetensors", blobs["model.safetensors"]), _f("config.json", blobs["config.json"], lfs=False)]
    monkeypatch.setattr(am, "hf_url", lambda repo, p, rev: f"https://huggingface.co/{repo}/resolve/{rev}/{p}")
    real = am.fetch
    monkeypatch.setattr(am, "fetch", lambda url, dest, size, stop, tick: real(url, dest, size, stop, tick, FakeOpener(blobs)))
    conv = []

    def fake_convert(runtime, src, out, bits):
        assert sorted(os.listdir(src)) == ["config.json", "model.safetensors"]
        conv.append((runtime, bits))
        os.makedirs(out)
    monkeypatch.setattr(am, "convert", fake_convert)
    mgr = am.Manager(str(tmp_path), _manifest(files))
    assert mgr.start("m", "8")
    st = _wait(mgr)
    assert st["status"] == "ready", st
    assert conv == [("mlx_lm", 8)] and mgr.installed("m").endswith("mlx-8bit")
    assert not os.path.exists(tmp_path / "m" / "original")   # originals deleted after conversion
    with pytest.raises(ValueError):
        mgr.start("m", "2")   # never below 4-bit
    assert mgr.delete("m") and not os.path.exists(tmp_path / "m") and mgr.installed("m") is None


def test_checksum_mismatch_is_refused_and_disk_is_checked(tmp_path, monkeypatch):
    blobs = {"model.safetensors": b"tampered"}
    files = [_f("model.safetensors", b"original")]
    files[0]["size"] = len(b"tampered")
    monkeypatch.setattr(am, "hf_url", lambda repo, p, rev: f"https://huggingface.co/{p}")
    real = am.fetch
    monkeypatch.setattr(am, "fetch", lambda url, dest, size, stop, tick: real(url, dest, size, stop, tick, FakeOpener(blobs)))
    mgr = am.Manager(str(tmp_path), _manifest(files))
    mgr.start("m")
    st = _wait(mgr)
    assert st["status"] == "error" and "checksum" in st["error"]
    assert not os.path.exists(tmp_path / "m" / "original" / "model.safetensors")
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(am.shutil, "disk_usage", lambda p: DU(1, 1, 10))
    mgr.start("m")
    st = _wait(mgr)
    assert st["status"] == "error" and "free disk" in st["error"]


def test_check_for_new_models_repins_only_on_request(tmp_path):
    files = [_f("model.safetensors", b"x")]
    mgr = am.Manager(str(tmp_path), None)
    mgr._manifest = None
    import json
    (tmp_path / "ai_models.json").write_text(json.dumps(_manifest(files)))
    changed = mgr.check_updates(pin=lambda repo, only=None: {"repo": repo, "revision": "def", "files": files})
    assert changed == ["M"] and mgr.model("m")["revision"] == "def"


def test_bundled_manifest_is_pinned():
    import json
    man = json.load(open(am.BUNDLED))
    ids = {m["id"]: m for m in man["models"]}
    assert {"medpsy-4b", "qwen3.5-9b", "qwen3.5-4b"} <= set(ids)
    assert ids["medpsy-4b"]["repo"] == "qvac/MedPsy-4B" and ids["qwen3.5-9b"]["repo"] == "Qwen/Qwen3.5-9B"
    for m in man["models"]:
        assert len(m["revision"]) == 40 and all(f["sha256"] or f["blob"] for f in m["files"])
        assert min(m["bits"]) >= 4
        assert all(len(f["sha256"]) == 64 for f in m["files"] if f["size"] > 10_000_000)


def test_one_tap_setup_picks_by_mac_installs_engine_and_offers_the_smaller_copy(tmp_path, monkeypatch):
    Disk = namedtuple("Disk", "total used free")
    mgr = am.Manager(str(tmp_path))
    started, have = [], {"mlx": False}
    monkeypatch.setattr(mgr, "start", lambda mid, v: started.append((mid, v)) or True)
    s = am.Setup(mgr, lambda kind: have[kind], install=lambda kind: have.update({kind: True}), silicon=lambda: True, ram=lambda: 16)
    assert s.plan()["id"] == "qwen3.5-4b" and s.plan()["vision"] and s.status()["step"] == "idle"
    assert am.Setup(mgr, None, silicon=lambda: True, ram=lambda: 32).plan()["id"] == "qwen3.5-9b"
    assert am.Setup(mgr, None, silicon=lambda: False, ram=lambda: 16).plan()["runtime"] == "llama"   # Intel / Linux
    monkeypatch.setattr(am.shutil, "disk_usage", lambda p: Disk(0, 0, 8e9))   # fits only the ready-made copy
    s.start()
    _until(lambda: s.step == "error")
    assert s.status()["fix"] == "community" and started == [] and not have["mlx"]
    s.start("prebuilt")
    _until(lambda: started)
    assert started == [("qwen3.5-4b", "prebuilt")] and have["mlx"]   # engine added in-app, then the download
    mgr.state["qwen3.5-4b"] = {"status": "downloading", "variant": "prebuilt", "done": 5e8, "total": 2e9, "t0": time.time() - 10, "done0": 0}
    st = s.status()
    assert st["step"] == "download" and st["pct"] == 25 and 25 <= st["etaS"] <= 35   # 1.5 GB left at 50 MB/s


def _until(cond):
    for _ in range(200):
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_download_errors_are_in_plain_words():
    assert "internet" in am.friendly("<urlopen error [Errno -2] Name or service not known>")
    assert "Free some space" in am.friendly("[Errno 28] No space left on device")

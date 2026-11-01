"""Settings > AI models: pinned Hugging Face downloads, checked file by file, then converted to MLX on the Mac.

Network use happens only here, and only after the user taps Download or 'Check for new models'.
Only HTTPS to huggingface.co and its CDN (*.hf.co) is allowed, including every redirect hop."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from urllib.parse import urlparse

BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models.json")
SKIP = {".gitattributes", ".gitignore"}
CHUNK = 1 << 20


def host_ok(url):
    p = urlparse(url)
    h = (p.hostname or "").lower()
    return p.scheme == "https" and (h in ("huggingface.co", "hf.co") or h.endswith((".huggingface.co", ".hf.co")))


class _Guard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not host_ok(newurl):
            raise OSError(f"Refused a redirect to {urlparse(newurl).hostname}: only huggingface.co and its CDN are allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPEN = urllib.request.build_opener(_Guard)


def hf_url(repo, path, revision):
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo, path, revision=revision)


def pin_repo(repo, revision=None, only=None):
    """Commit SHA + per-file size and hash from the Hub (LFS SHA256; small non-LFS files have only a git blob id)."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [{"path": s.rfilename, "size": s.size, "sha256": s.lfs.sha256 if s.lfs else None, "blob": s.blob_id}
             for s in info.siblings if s.rfilename not in SKIP and (not only or s.rfilename in only)]
    return {"repo": repo, "revision": info.sha, "files": files}


def file_ok(path, f):
    """SHA256 against the LFS hash; for small non-LFS files, the git blob SHA1 the Hub reports."""
    if f.get("sha256"):
        h = hashlib.sha256()
    else:
        h = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest() == (f.get("sha256") or f.get("blob"))


def fetch(url, dest, size, stop=lambda: False, progress=lambda n: None, opener=None):
    """Download to dest + '.part', resuming with a Range request. Returns False if paused."""
    if not host_ok(url):
        raise OSError(f"Refused {url}: only HTTPS to huggingface.co is allowed")
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with (opener or _OPEN).open(req, timeout=60) as r, open(part, "ab" if have else "wb") as out:
            if have and r.status != 206:   # server ignored Range: start over
                out.truncate(0)
                have = 0
            for b in iter(lambda: r.read(CHUNK), b""):
                out.write(b)
                have += len(b)
                progress(len(b))
                if stop():
                    return False
    os.replace(part, dest)
    return True


def convert(runtime, src, out, bits):
    """Local conversion to MLX (Apple silicon only); imported lazily so Linux and Intel Macs never load MLX."""
    if runtime == "mlx_vlm":
        from mlx_vlm import convert as conv
    else:
        from mlx_lm import convert as conv
    conv(hf_path=src, mlx_path=out, quantize=True, q_bits=bits)


class Manager:
    def __init__(self, root, manifest=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.override = os.path.join(root, "ai_models.json")   # written only by a 'Check for new models' tap
        self._manifest = manifest
        self.state_file = os.path.join(root, "state.json")
        try:
            with open(self.state_file) as fh:
                self.state = json.load(fh)
        except (OSError, ValueError):
            self.state = {}
        self.stops = {}
        self.lock = threading.Lock()

    # -- manifest --------------------------------------------------------
    def manifest(self):
        if self._manifest:
            return self._manifest
        path = self.override if os.path.exists(self.override) else BUNDLED
        with open(path) as fh:
            return json.load(fh)

    def model(self, mid):
        for m in self.manifest()["models"]:
            if m["id"] == mid:
                return m
        raise KeyError(mid)

    def check_updates(self, pin=pin_repo):
        """User tap only: re-pin every repo at its newest commit; the new pins become the manifest."""
        man = json.loads(json.dumps(self.manifest()))
        changed = []
        for m in man["models"]:
            only = [f["path"] for f in m["files"]] if m["runtime"] == "llama_cpp" else None
            new = pin(m["repo"], only=only)
            if new["revision"] != m["revision"]:
                changed.append(m["name"])
                m.update(new)
            if m.get("prebuilt"):
                p = pin(m["prebuilt"]["repo"])
                if p["revision"] != m["prebuilt"]["revision"]:
                    changed.append(m["name"] + " (community conversion)")
                    m["prebuilt"] = p
        if changed:
            with open(self.override + ".tmp", "w") as fh:
                json.dump(man, fh, indent=1)
            os.replace(self.override + ".tmp", self.override)
        return changed

    # -- state -----------------------------------------------------------
    def _save(self):
        with open(self.state_file + ".tmp", "w") as fh:
            json.dump(self.state, fh)
        os.replace(self.state_file + ".tmp", self.state_file)

    def _set(self, mid, **kw):
        with self.lock:
            self.state.setdefault(mid, {}).update(kw)
            self._save()

    def installed(self, mid):
        """Folder (MLX) or file (GGUF) to load, or None."""
        st = self.state.get(mid, {})
        p = st.get("path")
        return p if st.get("status") == "ready" and p and os.path.exists(p) else None

    def eta(self, mid):
        """Seconds left for the download (from this session's speed) plus a rough conversion estimate."""
        st, m = self.state.get(mid, {}), self.model(mid)
        conv = 0 if st.get("variant") == "prebuilt" or m["runtime"] == "llama_cpp" else m["diskGB"]["4"] * 25
        if st.get("status") == "converting":
            return conv
        left, dt = st.get("total", 0) - st.get("done", 0), time.time() - st.get("t0", time.time())
        speed = (st.get("done", 0) - st.get("done0", 0)) / dt if dt > 3 else 0
        return round(left / speed + conv) if speed > 0 else None

    def cards(self):
        out = []
        for m in self.manifest()["models"]:
            st = dict(self.state.get(m["id"], {"status": "idle"}))
            if st.get("status") == "ready" and not self.installed(m["id"]):
                st = {"status": "idle"}
            out.append({k: m.get(k) for k in ("id", "name", "purpose", "repo", "revision", "runtime", "bits", "ramGB",
                                               "diskGB", "license", "default", "vision")}
                       | {"downloadGB": round(sum(f["size"] for f in m["files"]) / 1e9, 1),
                          "prebuilt": m.get("prebuilt") and {"repo": m["prebuilt"]["repo"],
                                                            "downloadGB": round(sum(f["size"] for f in m["prebuilt"]["files"]) / 1e9, 1)},
                          "state": dict(st, etaS=self.eta(m["id"]) if st.get("status") in ("downloading", "converting") else None)})
        return out

    # -- actions ---------------------------------------------------------
    def start(self, mid, variant="4"):
        m = self.model(mid)
        if variant != "prebuilt" and int(variant) not in m["bits"]:
            raise ValueError("unknown variant")
        if variant == "prebuilt" and not m.get("prebuilt"):
            raise ValueError("no community conversion for this model")
        if self.state.get(mid, {}).get("status") in ("downloading", "converting"):
            return False
        self.stops[mid] = threading.Event()
        self._set(mid, status="downloading", variant=variant, error=None, msg="Starting")
        threading.Thread(target=self._work, args=(m, variant, self.stops[mid]), daemon=True).start()
        return True

    def pause(self, mid):
        if mid in self.stops:
            self.stops[mid].set()

    def delete(self, mid):
        self.model(mid)   # only known ids reach rmtree
        if self.state.get(mid, {}).get("status") in ("downloading", "converting"):
            return False
        shutil.rmtree(os.path.join(self.root, mid), ignore_errors=True)
        with self.lock:
            self.state.pop(mid, None)
            self._save()
        return True

    def resume_unfinished(self):
        """Downloads keep going across app restarts: the server calls this at start."""
        for mid, st in list(self.state.items()):
            if st.get("status") in ("downloading", "converting"):
                st["status"] = "paused"
                try:
                    self.start(mid, st.get("variant", "4"))
                except (KeyError, ValueError):
                    pass

    def _work(self, m, variant, stop):
        mid = m["id"]
        try:
            pre = variant == "prebuilt"
            src_repo = m["prebuilt"] if pre else m
            files = src_repo["files"]
            base = os.path.join(self.root, mid)
            mlx = m["runtime"] != "llama_cpp"
            src = os.path.join(base, "community-4bit" if pre else ("gguf" if not mlx else "original"))
            out = os.path.join(base, f"mlx-{variant}bit") if mlx and not pre else src
            total = sum(f["size"] for f in files)
            done = sum(os.path.getsize(os.path.join(src, f["path"])) for f in files
                       if os.path.exists(os.path.join(src, f["path"])))
            done += sum(os.path.getsize(os.path.join(src, f["path"]) + ".part") for f in files
                        if os.path.exists(os.path.join(src, f["path"]) + ".part"))
            need = (total - done) + (m["diskGB"][variant if not pre else "4"] * 1e9 if out != src else 0) + 1e9
            free = shutil.disk_usage(self.root).free
            if free < need:
                raise OSError(f"Needs {need / 1e9:.1f} GB free disk, only {free / 1e9:.1f} GB free")
            self._set(mid, total=total, done=done, t0=time.time(), done0=done)
            for f in files:
                dest = os.path.join(src, f["path"])
                if os.path.exists(dest):
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                self._set(mid, msg=f"Downloading {f['path']}")

                def tick(n):
                    self.state[mid]["done"] = self.state[mid].get("done", 0) + n
                if not fetch(hf_url(src_repo["repo"], f["path"], src_repo["revision"]), dest, f["size"], stop.is_set, tick):
                    self._set(mid, status="paused", msg="Paused")
                    return
                if not file_ok(dest, f):
                    os.remove(dest)
                    raise OSError(f"{f['path']} does not match the Hub's checksum; refused and deleted")
            if out != src:
                self._set(mid, status="converting", msg=f"Converting to MLX {variant}-bit on this Mac")
                shutil.rmtree(out, ignore_errors=True)
                convert(m["runtime"], src, out, int(variant))
                shutil.rmtree(src)   # originals are no longer needed
            path = out if mlx else os.path.join(src, next(f["path"] for f in files if f["path"].endswith(".gguf")))
            self._set(mid, status="ready", path=path, msg="Ready", revision=src_repo["revision"])
        except Exception as e:
            traceback.print_exc()
            self._set(mid, status="error", error=str(e), msg="Download failed")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = {"mlx": ["mlx-lm==0.32.2", "mlx-vlm==0.7.8"], "llama": ["llama-cpp-python==0.3.36"]}


def apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 16


def install_runtime(kind):
    """The AI engine (pinned packages) into the app's own Python, so no Terminal step is ever needed."""
    uv = os.path.join(ROOT, ".tools", "uv", "uv")
    cmd = ([uv, "pip", "install", "--python", sys.executable] if os.path.exists(uv) else [sys.executable, "-m", "pip", "install"])
    r = subprocess.run(cmd + RUNTIME[kind], capture_output=True, text=True)
    if r.returncode:
        raise OSError("Couldn't install the AI engine: " + ((r.stderr or r.stdout).strip().splitlines() or ["unknown error"])[-1])
    importlib.invalidate_caches()


class Setup:
    """'Private, on this Mac' in one tap: pick the model for this Mac, add the engine, download, convert."""

    def __init__(self, mgr, runtime_ok, install=install_runtime, silicon=apple_silicon, ram=ram_gb):
        self.mgr, self.runtime_ok, self.install, self.silicon, self.ram = mgr, runtime_ok, install, silicon, ram
        self.step, self.error, self.fix = None, None, None

    def plan(self):
        mid, kind = (("qwen3.5-9b" if self.ram() >= 24 else "qwen3.5-4b"), "mlx") if self.silicon() else ("medpsy-4b-gguf", "llama")
        m = self.mgr.model(mid)
        return {"id": mid, "name": m["name"], "vision": bool(m.get("vision")), "runtime": kind, "ramGB": round(self.ram()),
                "downloadGB": round(sum(f["size"] for f in m["files"]) / 1e9, 1), "diskGB": m["diskGB"]["4"],
                "prebuiltGB": m.get("prebuilt") and round(sum(f["size"] for f in m["prebuilt"]["files"]) / 1e9, 1)}

    def start(self, variant="4"):
        if self.step in ("runtime", "starting"):
            return False
        self.step, self.error, self.fix = "starting", None, None
        threading.Thread(target=self._work, args=(variant,), daemon=True).start()
        return True

    def _work(self, variant):
        p = self.plan()
        variant = self.mgr.state.get(p["id"], {}).get("variant") or variant   # Resume keeps the copy the user chose
        try:
            need = (p["prebuiltGB"] if variant == "prebuilt" else p["downloadGB"] + (p["diskGB"] if p["runtime"] == "mlx" else 0)) + 1
            free = shutil.disk_usage(self.mgr.root).free / 1e9
            if free < need:
                if variant != "prebuilt" and p["prebuiltGB"] and free >= p["prebuiltGB"] + 1:
                    raise _Fixable(f"Not enough free disk: setup needs {need:.0f} GB, {free:.0f} GB is free. "
                                   f"A smaller ready-made copy needs only {p['prebuiltGB'] + 1:.0f} GB.", "community")
                raise _Fixable(f"Not enough free disk: setup needs {need:.0f} GB, {free:.0f} GB is free. Free up space and try again.", "retry")
            if not self.runtime_ok(p["runtime"]):
                self.step = "runtime"
                self.install(p["runtime"])
                if not self.runtime_ok(p["runtime"]):
                    raise _Fixable("The AI engine was installed but can't start. Quit BodyGlass and open it again.", None)
            self.step = None
            if not self.mgr.installed(p["id"]):
                self.mgr.start(p["id"], variant)
        except _Fixable as e:
            self.step, self.error, self.fix = "error", str(e), e.fix
        except Exception as e:
            traceback.print_exc()
            self.step, self.error, self.fix = "error", str(e), "retry"

    def status(self):
        p = self.plan()
        if self.step in ("error", "runtime", "starting"):
            return {"step": self.step, "error": self.error, "fix": self.fix, "plan": p}
        st = self.mgr.state.get(p["id"], {})
        step = {"downloading": "download", "converting": "convert", "paused": "paused", "ready": "ready",
                "error": "error"}.get(st.get("status"), "idle")
        if step == "ready" and not self.mgr.installed(p["id"]):
            step = "idle"
        return {"step": step, "plan": p, "pct": int(100 * st.get("done", 0) / st["total"]) if st.get("total") else 0,
                "etaS": self.mgr.eta(p["id"]) if step in ("download", "convert") else None,
                "doneGB": round(st.get("done", 0) / 1e9, 1), "totalGB": round(st.get("total", 0) / 1e9, 1),
                "error": friendly(st.get("error")) if step == "error" else None, "fix": "retry" if step == "error" else None}


def friendly(err):
    """Download errors in plain words (the raw text stays in the log)."""
    e = str(err or "")
    if any(k in e for k in ("urlopen", "Name or service", "nodename", "timed out", "Connection", "Network is unreachable")):
        return "Can't reach the download server. Check your internet connection, then tap Try again: it carries on where it stopped."
    if "No space" in e or "disk" in e.lower():
        return "The disk filled up during setup. Free some space, then tap Try again."
    return e or "Setup stopped."


class _Fixable(Exception):
    def __init__(self, msg, fix):
        super().__init__(msg)
        self.fix = fix


if __name__ == "__main__":   # hidden power-user path (install.sh --with-ai-models): same checked download as Ask AI > Set up
    import sys
    import time
    from .server import app_dir
    mgr = Manager(os.path.join(os.environ.get("BODYGLASS_HOME") or app_dir(), "models"))
    mid = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-4b"
    if not mgr.installed(mid):
        mgr.start(mid, sys.argv[2] if len(sys.argv) > 2 else "4")
        while mgr.state[mid]["status"] in ("downloading", "converting"):
            st = mgr.state[mid]
            print(f"\r{st.get('msg', '')[:60]:60s} {100 * st.get('done', 0) // max(st.get('total', 1), 1):3d}%", end="", flush=True)
            time.sleep(2)
        print()
    st = mgr.state[mid]
    print("OK  " + mid + " ready" if st["status"] == "ready" else "Note: " + str(st.get("error")))
    sys.exit(0 if st["status"] == "ready" else 1)

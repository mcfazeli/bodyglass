"""Tiny local-only HTTP server: static viewer + processed cases + a few JSON endpoints.

Binds to 127.0.0.1 only. Nothing is ever uploaded anywhere.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import subprocess
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import ai_chat, ai_models, airclean, export3d, manifest_io, neck, phase2, pipeline, sequences, siwc, veins
from .dicom_scan import pick_t1, scan_folder
from .jobs import Busy, Jobs

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example")

BUSY = "This scan is already being processed. Please wait until it has finished."
DELETING = "This model is being deleted."
WEB = os.path.join(os.path.dirname(__file__), "web")


def app_dir():
    """Per-user data folder (saved models, AI models, ChatGPT sign-in)."""
    if platform.system() == "Darwin":
        return os.path.expanduser("~/Library/Application Support/BodyGlass")
    return os.path.expanduser("~/.local/share/bodyglass")


def default_cases_root():
    return os.environ.get("BODYGLASS_CASES") or os.path.join(app_dir(), "cases")


def start_processing(jobs, folder, cases_root, uid=None, then=None):
    """Queue the quick 3D model of a folder's T1 (or of series uid) and the work that follows it; returns the
    case id. Raises jobs.Busy if that case is already being built, ValueError if the folder has no 3D T1."""
    if uid is None:
        best = pick_t1(scan_folder(folder))
        if best is None:
            raise ValueError("No 3D T1-weighted series found in this folder.")
        uid = best.uid
    cid = pipeline.case_id_for(uid)
    case_dir = os.path.join(cases_root, cid)

    def phase2_then(up):
        run_phase2(case_dir, up)
        if then:   # e.g. the neck part of the same 'Add scan' folder, after the structures as before
            then(case_dir)

    def work(up):
        pipeline.run(folder, cases_root, up, uid=uid)
        # queued while this job still counts as running, so a deletion cannot slip in between
        jobs.quietly(cid, "sequences", lambda up: run_sequences(case_dir, up), "align")
        jobs.quietly(cid, "phase2", phase2_then, "heavy")

    jobs.submit(cid, "phase1", work, "heavy", new_case=True)
    return cid


def read_manifest(cases_root, cid):
    path = os.path.join(cases_root, cid, "manifest.json")
    if not cid.isalnum() or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def source_folder(case_dir, m):
    """Folder the case was built from; older cases did not store it, so use the folder holding their DICOM files."""
    if m.get("sourceFolder"):
        return m["sourceFolder"]
    try:
        with open(os.path.join(case_dir, "series.json")) as fh:
            sj = json.load(fh)
        files = [f for v in sj.values() for f in v.get("files", [])]
        return os.path.commonpath(files) if files else None
    except (OSError, ValueError):
        return None


def run_phase2(case_dir, up):
    """Labelled structures (FastSurfer) after the quick Phase 1 model is already on screen."""
    if os.environ.get("BODYGLASS_PHASE2", "1") == "0" or not phase2.available():
        return up(status="unavailable", msg="Structure labelling is not installed")
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        m = json.load(fh)
    if m.get("structures"):
        if not m.get("labelVolume"):  # case from an older version: add the slice-overlay labels
            try:
                phase2.backfill_labels(case_dir)
            except Exception:
                traceback.print_exc()
        return up(msg="Structures ready")
    up(1, "Labelling structures")
    try:
        phase2.build(case_dir, up, device=os.environ.get("BODYGLASS_DEVICE", "auto"))
        up(100, "Structures ready")
    except Exception as e:
        traceback.print_exc()
        up(status="error", error=str(e), msg="Structure labelling failed; the basic model still works")


def run_sequences(case_dir, up):
    if not os.path.exists(os.path.join(case_dir, "series.json")):
        return up(status="none", msg="No other scans in this folder")
    up(1, "Aligning other scans")
    try:
        out = sequences.build(case_dir, lambda pct, msg: up(pct * 0.8, msg))
        if any(q.get("kind") in ("swi", "mip") for q in out):
            up(80, "Tracing veins from the SWI scan")
            try:
                veins.build(case_dir)
            except Exception:
                traceback.print_exc()
        up(100, f"{len(out)} scans aligned" if out else "No other scans", **({} if out else {"status": "none"}))
    except Exception as e:
        traceback.print_exc()
        up(status="error", error=str(e), msg="Aligning the other scans failed; the T1 view still works")


def neck_work(case_dir, folder, uid=None):
    """The neck exam (second exam) of a saved case, as a job."""
    def work(up):
        try:
            info = neck.build(case_dir, folder, up, uid=uid)
            up(100, "Neck ready" if info["status"] == "ok" else "Neck exam could not be lined up")
        except Exception as e:
            traceback.print_exc()
            up(status="error", error=str(e), msg="Neck labelling failed; the brain model still works")
    return work


def add_scan(jobs, folder, cases_root, cid=None):
    """'Add scan': sort a folder (brain exam, neck exam, or a whole export with both) and start the right work.
    Returns (reply, http code). shortcut: headers are read here and again by the pipeline (a few seconds)."""
    r = neck.route(series := scan_folder(folder))
    brain, nk = r["brain"], (r["neck"] if neck.available() else None)
    case_dir = os.path.join(cases_root, cid) if cid and read_manifest(cases_root, cid) else None
    desc = lambda s: s.meta.get("SeriesDescription") or "scan"
    if brain:
        then = (lambda cd: jobs.quietly(os.path.basename(cd), "neck", neck_work(cd, folder, nk.uid), "heavy")) if nk else None
        try:
            new = start_processing(jobs, folder, cases_root, uid=brain.uid, then=then)
        except Busy:
            return {"error": BUSY}, 409
        return {"route": "both" if nk else "brain", "brain": desc(brain), "neck": nk and desc(nk), "caseId": new}, 200
    if r["neck"] and not nk:
        return {"error": "This folder holds a neck exam, but neck labelling is not installed. Run ./install.sh again to add it."}, 400
    if nk:
        if not case_dir:
            return {"error": "This folder holds a neck exam. Open the head model it belongs to first (My body), then add it."}, 400
        try:
            jobs.submit(cid, "neck", neck_work(case_dir, folder, nk.uid), "heavy")
        except Busy:
            return {"error": BUSY}, 409
        return {"route": "neck", "neck": desc(nk), "caseId": cid}, 200
    mf = {"multi-frame file without slice positions", "damaged or oversized multi-frame file"}
    hint = (" It holds a multi-frame DICOM file BodyGlass can't read ('Which scans are used?' says why)."
            if any(s.excluded in mf for s in series) else "")
    hint = next((" " + s.meta["unsupported"] for s in series if s.meta.get("unsupported")), hint)
    return {"error": "No usable scan found in this folder. BodyGlass needs a 3D T1 head scan or a sagittal neck scan "
                     "(see 'Which scans are used?' for what each kind is good for)." + hint}, 400


def case_summary(cid, m):
    """One 'My body' library card: a person = a case; its exams are the head scan and the linked neck exam."""
    exams = [{"kind": "Head MRI", "description": m["series"]["chosen"]["description"], "date": m.get("studyDate")}]
    if m.get("neck"):
        exams.append({"kind": "Neck MRI", "description": (m["neck"].get("series") or {}).get("description", ""),
                      "ok": m["neck"].get("status") == "ok"})
    return {"caseId": cid, "createdAt": m.get("createdAt"), "description": m["series"]["chosen"]["description"],
            "name": (m.get("person") or {}).get("name") or "", "exams": exams, "structures": bool(m.get("structures"))}


def choose_folder_macos(prompt="Choose your DICOM folder (it stays on this Mac)"):
    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() or None


def make_handler(cases_root, jobs):
    # AI chat data: ~/Library/Application Support/BodyGlass/{models,chatgpt} (BODYGLASS_HOME moves it, e.g. for tests)
    ai_root = os.environ.get("BODYGLASS_HOME") or app_dir()
    models = ai_models.Manager(os.path.join(ai_root, "models"))
    chatgpt = siwc.ChatGPT(os.path.join(ai_root, "chatgpt"))
    setup = ai_models.Setup(models, lambda kind: ai_chat.mlx_ok() if kind == "mlx" else ai_chat.llama_ok())

    class H(SimpleHTTPRequestHandler):
        ai = {"models": models, "chatgpt": chatgpt, "setup": setup,
              "backends": {"local": ai_chat.Local(models), "chatgpt": ai_chat.Cloud(chatgpt)}}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=WEB, **kw)

        def log_message(self, fmt, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            # the page may only talk to this local server
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self'; connect-src 'self' blob: data:; worker-src 'self' blob:")
            super().end_headers()

        def _trusted(self):
            """Only this app's own page: blocks DNS rebinding (Host) and cross-site POSTs (Origin)."""
            host = self.headers.get("Host", "")
            port = self.server.server_address[1]
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return False
            return self.command != "POST" or self.headers.get("Origin") in (None, f"http://{host}")

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _start(self, folder, reply, uid=None):
            try:
                return self._json(dict(reply, caseId=start_processing(jobs, folder, cases_root, uid=uid)))
            except Busy:
                return self._json({"error": BUSY}, 409)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)

        def translate_path(self, path):
            p = unquote(urlparse(path).path)
            if p.startswith("/example/"):
                rel = os.path.normpath(p[len("/example/"):]).lstrip("/")
                if rel.startswith(".."):
                    return os.path.join(WEB, "404")
                return os.path.join(EXAMPLE, rel)
            if p.startswith("/cases/"):
                rel = os.path.normpath(p[len("/cases/"):]).lstrip("/")
                if rel.startswith(".."):
                    return os.path.join(WEB, "404")
                return os.path.join(cases_root, rel)
            return super().translate_path(path)

        def do_GET(self):
            if not self._trusted():
                return self.send_error(403)
            p = urlparse(self.path).path
            if p == "/api/state":
                return self._json(dict(jobs=jobs.snapshot(), mac=platform.system() == "Darwin", neckAvailable=neck.available(), casesRoot=cases_root))
            if p == "/api/example":
                ej = os.path.join(EXAMPLE, "example.json")
                if not os.path.exists(ej):
                    return self._json({"available": False})
                with open(ej) as fh:
                    return self._json(dict(json.load(fh), available=True))
            if p == "/api/ai/state":   # no network: local files and the Keychain only
                A = self.ai
                loc = A["backends"]["local"]
                cards = A["models"].cards()
                for c in cards:
                    c["usable"] = loc.usable(c["id"])
                return self._json({"chatgpt": A["chatgpt"].status(), "models": cards, "setup": A["setup"].status(),
                                   "mlx": ai_chat.mlx_ok(), "llama": ai_chat.llama_ok(), "modelsDir": A["models"].root})
            if p == "/api/ai/chatgpt-models":   # only asked for once the user picked ChatGPT and is signed in
                try:
                    return self._json({"models": self.ai["chatgpt"].models()})
                except (siwc.ChatGPTError, OSError) as e:
                    return self._json({"error": str(e)}, 400)
            if p == "/api/cases":
                out = []
                if os.path.isdir(cases_root):
                    for cid in sorted(os.listdir(cases_root)):
                        mf = os.path.join(cases_root, cid, "manifest.json")
                        try:
                            with open(mf) as fh:
                                m = json.load(fh)
                            out.append(case_summary(cid, m))
                        except (OSError, ValueError, KeyError):
                            continue   # missing or damaged case: leave it out, keep the list working
                return self._json(out)
            return super().do_GET()

        def do_POST(self):
            if not self._trusted():
                return self.send_error(403)
            p = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
            if p == "/api/choose-folder":
                if platform.system() != "Darwin":
                    return self._json({"error": "Folder dialog is macOS only; pass the folder on the command line."}, 400)
                folder = choose_folder_macos()
                if not folder:
                    return self._json({"folder": None})
                return self._start(folder, {"folder": folder})
            if p == "/api/export":
                cid = str(data.get("caseId", ""))
                case_dir = os.path.join(cases_root, cid)
                fmt = "3mf" if data.get("format") == "3mf" else "stl"
                if not cid.isalnum() or not os.path.isdir(case_dir):
                    return self._json({"error": "case not found"}, 400)
                try:
                    blob, info = export3d.export(case_dir, list(data.get("ids") or []), fmt)
                except Exception as e:
                    traceback.print_exc()
                    return self._json({"error": str(e)}, 400)
                self.send_response(200)
                self.send_header("Content-Type", "model/3mf" if fmt == "3mf" else "model/stl")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("X-Export-Info", json.dumps(info, ensure_ascii=True))
                self.end_headers()
                self.wfile.write(blob)
                return
            if p == "/api/say":   # optional narration with the Mac's built-in offline voice
                if platform.system() != "Darwin":
                    return self._json({"error": "speech is macOS only"}, 400)
                text = str(data.get("text", ""))[:1500]
                subprocess.run(["killall", "say"], capture_output=True)
                if text:
                    subprocess.Popen(["say", "-r", "185", text])
                return self._json({"ok": True})
            if p == "/api/neck":   # folder from the Mac's folder dialog, or given (command line / tests)
                cid = str(data.get("caseId", ""))
                case_dir = os.path.join(cases_root, cid)
                if not cid.isalnum() or not os.path.exists(os.path.join(case_dir, "manifest.json")):
                    return self._json({"error": "case not found"}, 400)
                if not neck.available():
                    return self._json({"error": "Neck labelling is not installed. Run ./install.sh again to add it."}, 400)
                folder = data.get("folder")
                if not folder:
                    if platform.system() != "Darwin":
                        return self._json({"error": "Folder dialog is macOS only."}, 400)
                    folder = choose_folder_macos("Choose the folder of your neck MRI (it stays on this Mac)")
                    if not folder:
                        return self._json({"folder": None})
                if not os.path.isdir(folder):
                    return self._json({"error": "folder not found"}, 400)
                try:
                    jobs.submit(cid, "neck", neck_work(case_dir, folder), "heavy")
                except Busy:
                    return self._json({"error": BUSY}, 409)
                return self._json({"folder": folder, "caseId": cid})
            if p == "/api/rebuild":   # 'Use for 3D instead': another series of the same exam -> a new model
                cid, uid = str(data.get("caseId", "")), str(data.get("seriesUid", ""))
                m = read_manifest(cases_root, cid)
                if not m or uid not in [s["uid"] for s in m["series"]["all"]]:
                    return self._json({"error": "case or series not found"}, 400)
                folder = source_folder(os.path.join(cases_root, cid), m)
                if not folder or not os.path.isdir(folder):
                    return self._json({"error": "The original scan folder has moved. Use Add scan and choose it again."}, 400)
                return self._start(folder, {"ok": True}, uid)
            if p == "/api/reskin":   # skin-only rebuild with the mouth/nostril closing (older cases)
                cid = str(data.get("caseId", ""))
                try:
                    with jobs.writing(cid):
                        if not read_manifest(cases_root, cid):
                            return self._json({"error": "case not found"}, 400)
                        pipeline.reskin(os.path.join(cases_root, cid))
                except Busy:
                    return self._json({"error": DELETING}, 409)
                except Exception as e:
                    traceback.print_exc()
                    return self._json({"error": f"Could not update the skin: {e}"}, 400)
                return self._json({"ok": True})
            if p == "/api/add":   # one 'Add scan' button: brain, neck or both, sorted automatically
                folder = data.get("folder")
                if not folder:
                    if platform.system() != "Darwin":
                        return self._json({"error": "Folder dialog is macOS only; pass the folder on the command line."}, 400)
                    folder = choose_folder_macos("Choose a scan folder: head or neck MRI, or a whole export (it stays on this Mac)")
                    if not folder:
                        return self._json({"folder": None})
                if not os.path.isdir(folder):
                    return self._json({"error": "folder not found"}, 400)
                try:
                    reply, code = add_scan(jobs, folder, cases_root, str(data.get("caseId") or ""))
                except Exception as e:
                    traceback.print_exc()
                    return self._json({"error": f"Could not read this folder: {e}"}, 400)
                return self._json(dict(reply, folder=folder), code)
            if p in ("/api/case/rename", "/api/case/delete", "/api/reveal"):   # 'My body' library
                cid = str(data.get("caseId") or "")
                if p == "/api/reveal" and not cid:
                    path = cases_root
                else:
                    if not read_manifest(cases_root, cid):
                        return self._json({"error": "case not found"}, 400)
                    path = os.path.join(cases_root, cid)
                try:
                    if p == "/api/case/delete":   # the 3D model only; the user's DICOM folder is never touched
                        real = os.path.realpath(path)
                        if os.path.dirname(real) != os.path.realpath(cases_root):
                            return self._json({"error": "case not found"}, 400)
                        jobs.delete(cid, lambda: os.path.isdir(real) and shutil.rmtree(real))
                    elif p == "/api/case/rename" or platform.system() == "Darwin":
                        with jobs.writing(cid) if cid else contextlib.nullcontext():
                            if cid and not read_manifest(cases_root, cid):   # deleted meanwhile: never recreate it
                                return self._json({"error": "case not found"}, 400)
                            if p == "/api/case/rename":
                                manifest_io.update(path, {"person": {"name": str(data.get("name", "")).strip()[:80]}})
                            else:
                                os.makedirs(path, exist_ok=True)
                                subprocess.run(["open", "-R", path] if cid else ["open", path])
                except Busy:
                    return self._json({"error": DELETING if cid in jobs.deleting else
                                       "This model is still being worked on. Try again when it is finished."}, 409)
                return self._json({"ok": True, "path": path})
            if p.startswith("/api/ai/"):
                return self._ai_post(p, data)
            if p == "/api/open":
                folder = data.get("folder")
                if not folder or not os.path.isdir(folder):
                    return self._json({"error": "folder not found"}, 400)
                return self._start(folder, {"ok": True})
            self._json({"error": "not found"}, 404)

        def _stream(self, gen):
            """Answer as NDJSON lines while it is generated; a closed connection (Stop) ends the generator."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            text = []
            try:
                for ev in gen:
                    text.append(ev.get("t", ""))
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
                if ai_chat.needs_caveat("".join(text)):   # every backend: a verdict on the scan gets a one-line note
                    self._line({"caveat": ai_chat.CAVEAT})
                self.wfile.write(b'{"done": true}\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ai_chat.ChatError as e:
                self._line({"error": str(e), "fix": e.fix})
            except Exception as e:
                traceback.print_exc()
                self._line({"error": f"The AI stopped with an error: {e}", "fix": "retry"})
            finally:
                gen.close()

        def _line(self, ev):
            try:
                self.wfile.write((json.dumps(ev) + "\n").encode())
            except OSError:
                pass

        def _ai_post(self, p, data):
            A = self.ai
            try:
                if p in ("/api/ai/preview", "/api/ai/chat"):
                    payload = ai_chat.build_payload(data.get("ctx"), data.get("question"), data.get("history"), data.get("image"))
                    if p == "/api/ai/preview":
                        return self._json({"preview": ai_chat.preview(payload)})
                    b = A["backends"].get(data.get("backend"))
                    if not b:
                        return self._json({"error": "unknown AI backend"}, 400)
                    return self._stream(b.stream(payload, data.get("model") or None))
                if p == "/api/ai/setup":   # 'Private, on this Mac': one tap
                    return self._json({"started": A["setup"].start("prebuilt" if data.get("variant") == "prebuilt" else "4")})
                if p == "/api/ai/signin":
                    A["chatgpt"].start_signin()
                    return self._json({"ok": True})
                if p == "/api/ai/signout":
                    return self._json({"revoked": A["chatgpt"].sign_out()})
                M, mid = A["models"], str(data.get("id", ""))
                if p == "/api/ai/models/download":
                    return self._json({"started": M.start(mid, str(data.get("variant", "4")))})
                if p == "/api/ai/models/pause":
                    M.pause(mid)
                    return self._json({"ok": True})
                if p == "/api/ai/models/delete":
                    return self._json({"deleted": M.delete(mid)})
                if p == "/api/ai/models/check":   # the only manifest update path: a user tap
                    return self._json({"changed": M.check_updates()})
            except (ValueError, KeyError, RuntimeError, OSError, siwc.ChatGPTError) as e:
                traceback.print_exc()
                return self._json({"error": str(e) or "AI request failed"}, 400)
            return self._json({"error": "not found"}, 404)

    return H


def _backfill_all(jobs, cases_root):
    """Older cached cases get labels.u8 for the slice overlay, missing scan alignments and the air cleanup
    (seconds per case, from cached output). Queued as jobs so a case is never deleted mid-upgrade."""
    for cid in sorted(os.listdir(cases_root)):
        case_dir = os.path.join(cases_root, cid)

        def upgrade(up, case_dir=case_dir):
            with open(os.path.join(case_dir, "manifest.json")) as fh:
                m = json.load(fh)
            if m.get("structures") and not m.get("labelVolume"):
                phase2.backfill_labels(case_dir)
            airclean.backfill(case_dir)      # once per case
        jobs.quietly(cid, "upgrade", upgrade, "align")   # skipped for folders without a manifest
        if os.path.exists(os.path.join(case_dir, "series.json")):   # cached: only restores/finishes what is missing
            jobs.quietly(cid, "sequences", lambda up, case_dir=case_dir: run_sequences(case_dir, up), "align")


def serve(port=8765, cases_root=None):
    cases_root = cases_root or default_cases_root()
    os.makedirs(cases_root, exist_ok=True)
    jobs = Jobs(cases_root)
    _backfill_all(jobs, cases_root)
    handler = make_handler(cases_root, jobs)
    handler.ai["models"].resume_unfinished()   # a download the user started keeps going after a restart
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.jobs = jobs
    return httpd

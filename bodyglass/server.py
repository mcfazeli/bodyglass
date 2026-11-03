"""Tiny local-only HTTP server: static viewer + processed cases + a few JSON endpoints.

Binds to 127.0.0.1 only. Nothing is ever uploaded anywhere.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import ai_chat, ai_models, airclean, export3d, manifest_io, neck, phase2, pipeline, sequences, siwc, veins
from .dicom_scan import scan_folder

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example")

_START = threading.Lock()
BUSY = "Already processing a scan. Please wait until it has finished."
WEB = os.path.join(os.path.dirname(__file__), "web")
STATE = {"status": "idle", "pct": 0, "msg": "", "caseId": None, "error": None,
         "phase2": {"status": "idle", "pct": 0, "msg": "", "error": None},
         "sequences": {"status": "idle", "pct": 0, "msg": "", "error": None, "caseId": None},
         "neck": {"status": "idle", "pct": 0, "msg": "", "error": None, "caseId": None}}


def app_dir():
    """Per-user data folder (saved models, AI models, ChatGPT sign-in)."""
    if platform.system() == "Darwin":
        return os.path.expanduser("~/Library/Application Support/BodyGlass")
    return os.path.expanduser("~/.local/share/bodyglass")


def default_cases_root():
    return os.environ.get("BODYGLASS_CASES") or os.path.join(app_dir(), "cases")


def start_processing(folder, cases_root, uid=None, then=None):
    """Returns False if a scan is already being processed (one at a time: RAM and the shared STATE)."""
    # shortcut: only Phase 1 is guarded; a new scan's FastSurfer can overlap the previous one's (~2 x 4 GB), queue Phase 2 if 16 GB Macs need it
    with _START:
        if STATE["status"] == "running" or STATE["neck"]["status"] == "running":
            return False
        STATE.update(status="running", pct=0, msg="Starting", error=None, caseId=None)

    def prog(pct, msg):
        STATE.update(pct=pct, msg=msg)

    def work():
        try:
            cid = pipeline.run(folder, cases_root, prog, uid=uid)
            STATE.update(status="done", caseId=cid)
        except Exception as e:  # report to UI
            traceback.print_exc()
            STATE.update(status="error", error=str(e))
            return
        case_dir = os.path.join(cases_root, cid)
        # other scans are aligned alongside the structure labelling (both read the finished T1 grid)
        threading.Thread(target=run_sequences, args=(case_dir,), daemon=True).start()
        run_phase2(case_dir)
        if then:   # e.g. the neck part of the same 'Add scan' folder
            then(case_dir)

    threading.Thread(target=work, daemon=True).start()
    return True


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


def run_phase2(case_dir):
    """Labelled structures (FastSurfer) after the quick Phase 1 model is already on screen."""
    P = STATE["phase2"]
    if os.environ.get("BODYGLASS_PHASE2", "1") == "0" or not phase2.available():
        P.update(status="unavailable", msg="Structure labelling is not installed")
        return
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        m = json.load(fh)
    if m.get("structures"):
        if not m.get("labelVolume"):  # case from an older version: add the slice-overlay labels
            try:
                phase2.backfill_labels(case_dir)
            except Exception:
                traceback.print_exc()
        P.update(status="done", pct=100, msg="Structures ready")
        return
    P.update(status="running", pct=1, msg="Labelling structures", error=None)
    try:
        phase2.build(case_dir, lambda pct, msg: P.update(pct=pct, msg=msg),
                     device=os.environ.get("BODYGLASS_DEVICE", "auto"))
        P.update(status="done", pct=100, msg="Structures ready")
    except Exception as e:
        traceback.print_exc()
        P.update(status="error", error=str(e), msg="Structure labelling failed; the basic model still works")


def run_sequences(case_dir):
    S = STATE["sequences"]
    S.update(caseId=os.path.basename(case_dir), error=None)
    if not os.path.exists(os.path.join(case_dir, "series.json")):
        S.update(status="none", msg="No other scans in this folder")
        return
    S.update(status="running", pct=1, msg="Aligning other scans")
    try:
        out = sequences.build(case_dir, lambda pct, msg: S.update(pct=pct * 0.8, msg=msg))
        if any(q.get("kind") in ("swi", "mip") for q in out):
            S.update(pct=80, msg="Tracing veins from the SWI scan")
            try:
                veins.build(case_dir)
            except Exception:
                traceback.print_exc()
        S.update(status="done" if out else "none", pct=100, msg=f"{len(out)} scans aligned" if out else "No other scans")
    except Exception as e:
        traceback.print_exc()
        S.update(status="error", error=str(e), msg="Aligning the other scans failed; the T1 view still works")


def start_neck(case_dir, folder, uid=None):
    """Second exam (neck) for an opened case. False if a scan or neck exam is already being processed."""
    N = STATE["neck"]
    with _START:
        if STATE["status"] == "running" or N["status"] == "running":
            return False
        N.update(status="running", pct=0, msg="Starting", error=None, caseId=os.path.basename(case_dir))

    threading.Thread(target=run_neck, args=(case_dir, folder, uid), daemon=True).start()
    return True


def run_neck(case_dir, folder, uid=None):
    N = STATE["neck"]
    N.update(status="running", pct=0, msg="Starting", error=None, caseId=os.path.basename(case_dir))
    try:
        info = neck.build(case_dir, folder, lambda pct, msg: N.update(pct=pct, msg=msg), uid=uid)
        N.update(status="done", pct=100, msg="Neck ready" if info["status"] == "ok" else "Neck exam could not be lined up")
    except Exception as e:
        traceback.print_exc()
        N.update(status="error", error=str(e), msg="Neck labelling failed; the brain model still works")


def add_scan(folder, cases_root, cid=None):
    """'Add scan': sort a folder (brain exam, neck exam, or a whole export with both) and start the right work.
    Returns (reply, http code). shortcut: headers are read here and again by the pipeline (a few seconds)."""
    r = neck.route(series := scan_folder(folder))
    brain, nk = r["brain"], (r["neck"] if neck.available() else None)
    case_dir = os.path.join(cases_root, cid) if cid and read_manifest(cases_root, cid) else None
    desc = lambda s: s.meta.get("SeriesDescription") or "scan"
    if brain:
        then = (lambda cd: run_neck(cd, folder, nk.uid)) if nk else None
        if not start_processing(folder, cases_root, uid=brain.uid, then=then):
            return {"error": BUSY}, 409
        return {"route": "both" if nk else "brain", "brain": desc(brain), "neck": nk and desc(nk)}, 200
    if r["neck"] and not nk:
        return {"error": "This folder holds a neck exam, but neck labelling is not installed. Run ./install.sh again to add it."}, 400
    if nk:
        if not case_dir:
            return {"error": "This folder holds a neck exam. Open the head model it belongs to first (My body), then add it."}, 400
        if not start_neck(case_dir, folder, nk.uid):
            return {"error": BUSY}, 409
        return {"route": "neck", "neck": desc(nk)}, 200
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


def busy_with(cid):
    return any(S.get("status") == "running" and S.get("caseId") == cid
               for S in (STATE, STATE["neck"], STATE["phase2"], STATE["sequences"]))


def choose_folder_macos(prompt="Choose your DICOM folder (it stays on this Mac)"):
    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() or None


def make_handler(cases_root):
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
                return self._json(dict(STATE, mac=platform.system() == "Darwin", neckAvailable=neck.available(), casesRoot=cases_root))
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
                if STATE["status"] == "running":
                    return self._json({"error": BUSY}, 409)
                folder = choose_folder_macos()
                if folder and not start_processing(folder, cases_root):
                    return self._json({"error": BUSY}, 409)
                return self._json({"folder": folder})
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
                if not start_neck(case_dir, folder):
                    return self._json({"error": BUSY}, 409)
                return self._json({"folder": folder})
            if p == "/api/rebuild":   # 'Use for 3D instead': another series of the same exam -> a new model
                cid, uid = str(data.get("caseId", "")), str(data.get("seriesUid", ""))
                m = read_manifest(cases_root, cid)
                if not m or uid not in [s["uid"] for s in m["series"]["all"]]:
                    return self._json({"error": "case or series not found"}, 400)
                folder = source_folder(os.path.join(cases_root, cid), m)
                if not folder or not os.path.isdir(folder):
                    return self._json({"error": "The original scan folder has moved. Use Add scan and choose it again."}, 400)
                if not start_processing(folder, cases_root, uid=uid):
                    return self._json({"error": BUSY}, 409)
                return self._json({"ok": True})
            if p == "/api/reskin":   # skin-only rebuild with the mouth/nostril closing (older cases)
                cid = str(data.get("caseId", ""))
                if not read_manifest(cases_root, cid):
                    return self._json({"error": "case not found"}, 400)
                try:
                    pipeline.reskin(os.path.join(cases_root, cid))
                except Exception as e:
                    traceback.print_exc()
                    return self._json({"error": f"Could not update the skin: {e}"}, 400)
                return self._json({"ok": True})
            if p == "/api/add":   # one 'Add scan' button: brain, neck or both, sorted automatically
                folder = data.get("folder")
                if not folder:
                    if platform.system() != "Darwin":
                        return self._json({"error": "Folder dialog is macOS only; pass the folder on the command line."}, 400)
                    if STATE["status"] == "running" or STATE["neck"]["status"] == "running":
                        return self._json({"error": BUSY}, 409)
                    folder = choose_folder_macos("Choose a scan folder: head or neck MRI, or a whole export (it stays on this Mac)")
                    if not folder:
                        return self._json({"folder": None})
                if not os.path.isdir(folder):
                    return self._json({"error": "folder not found"}, 400)
                try:
                    reply, code = add_scan(folder, cases_root, str(data.get("caseId") or ""))
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
                if p == "/api/case/rename":
                    manifest_io.update(path, {"person": {"name": str(data.get("name", "")).strip()[:80]}})
                elif p == "/api/case/delete":   # the 3D model only; the user's DICOM folder is never touched
                    if busy_with(cid):
                        return self._json({"error": "This model is still being worked on. Try again when it is finished."}, 409)
                    real = os.path.realpath(path)
                    if os.path.dirname(real) != os.path.realpath(cases_root):
                        return self._json({"error": "case not found"}, 400)
                    shutil.rmtree(real)
                elif platform.system() == "Darwin":
                    os.makedirs(path, exist_ok=True)
                    subprocess.run(["open", "-R", path] if cid else ["open", path])
                return self._json({"ok": True, "path": path})
            if p.startswith("/api/ai/"):
                return self._ai_post(p, data)
            if p == "/api/open":
                folder = data.get("folder")
                if not folder or not os.path.isdir(folder):
                    return self._json({"error": "folder not found"}, 400)
                if not start_processing(folder, cases_root):
                    return self._json({"error": BUSY}, 409)
                return self._json({"ok": True})
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


def _backfill_all(cases_root):
    """Older cached cases get labels.u8 for the slice overlay (seconds per case, uses cached FastSurfer output)."""
    for cid in sorted(os.listdir(cases_root)):
        mf = os.path.join(cases_root, cid, "manifest.json")
        if not os.path.isfile(mf):
            continue
        try:
            with open(mf) as fh:
                m = json.load(fh)
            if m.get("structures") and not m.get("labelVolume"):
                phase2.backfill_labels(os.path.join(cases_root, cid))
            if os.path.exists(os.path.join(cases_root, cid, "series.json")):
                sequences.build(os.path.join(cases_root, cid))   # cached: only restores/finishes what is missing
                veins.build(os.path.join(cases_root, cid))       # cached
            airclean.backfill(os.path.join(cases_root, cid))      # once per case
        except Exception:
            traceback.print_exc()


def serve(port=8765, cases_root=None):
    cases_root = cases_root or default_cases_root()
    os.makedirs(cases_root, exist_ok=True)
    threading.Thread(target=_backfill_all, args=(cases_root,), daemon=True).start()
    handler = make_handler(cases_root)
    handler.ai["models"].resume_unfinished()   # a download the user started keeps going after a restart
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return httpd

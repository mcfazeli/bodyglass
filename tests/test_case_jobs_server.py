"""The server's job wiring: each case's stages in /api/state, deletion conflicts over HTTP, and a failed
detailed stage that leaves the quick model usable. A fake pipeline writes synthetic case files; no scans."""
import http.client
import json
import os
import threading
import time

from bodyglass import pipeline, server


def call(port, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST" if body is not None else "GET", path, body=json.dumps(body) if body is not None else None,
              headers={"Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def wait(cond, timeout=10):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def test_two_cases_queue_report_separately_and_delete_safely(tmp_path, monkeypatch):
    root, dicom = str(tmp_path / "cases"), tmp_path / "dicom"
    dicom.mkdir(); (dicom / "IM0001").write_bytes(b"synthetic")       # stands in for the user's scan folder
    gates = {}

    def fake_run(folder, cases_root, progress, uid):                   # Phase 1: blocks until the test releases it
        cid = pipeline.case_id_for(uid)
        progress(20, f"building {uid}")
        assert gates[uid].wait(10)
        os.makedirs(os.path.join(cases_root, cid), exist_ok=True)
        with open(os.path.join(cases_root, cid, "manifest.json"), "w") as fh:
            json.dump({"caseId": cid, "series": {"chosen": {"description": uid}, "all": []}}, fh)
        return cid

    def failing_phase2(case_dir, up, device):
        up(30, "labelling")
        raise RuntimeError("synthetic FastSurfer failure")
    monkeypatch.setattr(server.pipeline, "run", fake_run)
    monkeypatch.setattr(server.phase2, "available", lambda: True)
    monkeypatch.setattr(server.phase2, "build", failing_phase2)
    httpd = server.serve(0, root)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        for uid in ("1.2.A", "1.2.B"):
            gates[uid] = threading.Event()
        a = server.start_processing(httpd.jobs, str(dicom), root, uid="1.2.A")
        b = server.start_processing(httpd.jobs, str(dicom), root, uid="1.2.B")
        st = lambda: call(port, "/api/state")[1]["jobs"]
        wait(lambda: st()[a]["phase1"]["status"] == "running")
        s = st()
        assert s[a]["phase1"]["msg"] == "building 1.2.A" and s[b]["phase1"]["status"] == "queued"   # one heavy job at a time
        for cid in (a, b):                                                 # running, queued: refused, nothing removed
            code, r = call(port, "/api/case/delete", {"caseId": cid})
            assert code in (400, 409)                                     # 400: no manifest yet, 409: being worked on
        gates["1.2.A"].set()
        wait(lambda: st()[b]["phase1"]["status"] == "running")            # B's quick model comes before A's structures
        s = st()
        assert s[a]["phase1"]["status"] == "done" and s[a]["phase2"]["status"] == "queued"
        assert call(port, "/api/case/delete", {"caseId": a})[0] == 409    # A's follow-up work is pending
        gates["1.2.B"].set()
        wait(lambda: all(st()[c].get("phase2", {}).get("status") == "error" for c in (a, b)))   # failures free the slot
        s = st()
        assert s[a]["phase1"]["status"] == "done" and "basic model still works" in s[a]["phase2"]["msg"]
        assert call(port, f"/cases/{a}/manifest.json")[0] == 200          # the quick model is still usable
        gates["1.2.C"] = threading.Event()
        c = server.start_processing(httpd.jobs, str(dicom), root, uid="1.2.C")
        wait(lambda: st()[c]["phase1"]["status"] == "running")
        code, r = call(port, "/api/case/delete", {"caseId": a})           # idle while another case is being built
        assert code == 200 and not os.path.exists(os.path.join(root, a)) and (dicom / "IM0001").exists()
        assert a not in st() and st()[c]["phase1"]["msg"] == "building 1.2.C"
        gates["1.2.C"].set()
        wait(lambda: st()[c].get("phase2", {}).get("status") == "error")
        for cid in (b, c):
            assert call(port, "/api/case/delete", {"caseId": cid})[0] == 200
        time.sleep(0.3)
        assert os.listdir(root) == [] and (dicom / "IM0001").exists()     # nothing recreated after the deletions
    finally:
        for g in gates.values():
            g.set()
        httpd.shutdown()


def test_case_being_built_is_refused_twice_and_delete_conflict_is_409(tmp_path, monkeypatch):
    root = str(tmp_path / "cases")
    cid = pipeline.case_id_for("1.2.C")
    os.makedirs(os.path.join(root, cid))
    with open(os.path.join(root, cid, "manifest.json"), "w") as fh:     # an existing case being rebuilt
        json.dump({"caseId": cid, "series": {"chosen": {"description": "T1"}, "all": []}}, fh)
    gate = threading.Event()
    monkeypatch.setattr(server.pipeline, "run", lambda f, r, p, uid: gate.wait(10) and pipeline.case_id_for(uid))
    monkeypatch.setattr(server.phase2, "available", lambda: False)
    httpd = server.serve(0, root)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        server.start_processing(httpd.jobs, str(tmp_path), root, uid="1.2.C")
        try:
            server.start_processing(httpd.jobs, str(tmp_path), root, uid="1.2.C")
            raise AssertionError("queued the same case twice")
        except server.Busy:
            pass
        code, r = call(port, "/api/case/delete", {"caseId": cid})
        assert code == 409 and "still being worked on" in r["error"] and os.path.exists(os.path.join(root, cid))
        gate.set()
        wait(lambda: call(port, "/api/state")[1]["jobs"][cid].get("phase2", {}).get("status") == "unavailable")
        assert call(port, "/api/case/delete", {"caseId": cid})[0] == 200
    finally:
        gate.set()
        httpd.shutdown()

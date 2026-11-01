"""'My body' library routes: list with names/exams, rename, delete (model only, safe ids), Host/Origin still enforced."""
import http.client, json, os, threading

from bodyglass import server


def call(port, path, body=None, origin=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"127.0.0.1:{port}"}
    if origin:
        h["Origin"] = origin
    c.request("POST" if body is not None else "GET", path, body=json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    raw = r.read()
    return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw)


def case(root, cid, desc):
    os.makedirs(os.path.join(root, cid))
    with open(os.path.join(root, cid, "manifest.json"), "w") as fh:
        json.dump({"caseId": cid, "createdAt": "2026-10-18", "series": {"chosen": {"description": desc}, "all": []},
                   "neck": {"status": "ok", "series": {"description": "T2w"}}}, fh)


def test_library_list_rename_delete(tmp_path):
    root = str(tmp_path / "cases")
    case(root, "aaa111", "MPRAGE"); case(root, "bbb222", "T1 3D")
    outside = tmp_path / "keep"; outside.mkdir()                         # stands in for the user's DICOM folder
    httpd = server.serve(0, root)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        st, cases = call(port, "/api/cases")
        assert st == 200 and [c["caseId"] for c in cases] == ["aaa111", "bbb222"]
        assert [e["kind"] for e in cases[0]["exams"]] == ["Head MRI", "Neck MRI"]
        assert call(port, "/api/case/rename", {"caseId": "aaa111", "name": "  Me  "})[0] == 200
        assert call(port, "/api/cases")[1][0]["name"] == "Me"
        for bad in ("../keep", "", "nope", "aaa111/.."):
            assert call(port, "/api/case/delete", {"caseId": bad})[0] == 400
        assert call(port, "/api/case/delete", {"caseId": "bbb222"}, origin="https://evil.example")[0] == 403
        assert call(port, "/api/add", {"folder": str(outside)}, origin="https://evil.example")[0] == 403
        assert call(port, "/api/case/delete", {"caseId": "bbb222"})[0] == 200
        assert not os.path.exists(os.path.join(root, "bbb222")) and outside.exists() and os.path.exists(os.path.join(root, "aaa111"))
        st, r = call(port, "/api/add", {"folder": str(outside)})            # empty folder: plain reason, nothing started
        assert st == 400 and "No usable scan" in r["error"]
    finally:
        httpd.shutdown()

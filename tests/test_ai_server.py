"""AI routes on the local server: state, preview == what the backend gets, local chat with no egress, guards kept."""
import http.client
import json
import threading

from bodyglass import ai_chat, server
from test_ai_chat import no_egress  # noqa: F401  (fixture)
from test_library import call


def chat(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/api/ai/chat", body=json.dumps(body), headers={"Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    assert r.getheader("Content-Type") == "application/x-ndjson"
    return r.status, [json.loads(x) for x in r.read().decode().splitlines()]


def test_ai_routes_preview_chat_and_guards(tmp_path, no_egress, monkeypatch):  # noqa: F811
    monkeypatch.setenv("BODYGLASS_HOME", str(tmp_path))
    httpd = server.serve(0, str(tmp_path / "cases"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    got = []

    class Fake:
        def usable(self, mid): return False

        def stream(self, payload, want=None):
            got.append(payload)
            yield {"model": "fake", "note": None}
            yield {"t": "An "}
            if "fail" in payload["messages"][-1]["text"]:
                raise ai_chat.ChatError("Out of memory.", "lighter")
            yield {"t": "answer"}
            if "size" in payload["messages"][-1]["text"]:
                yield {"t": ". It looks normal."}
    httpd.RequestHandlerClass.ai["backends"]["local"] = Fake()
    try:
        st, s = call(port, "/api/ai/state")
        assert st == 200 and {m["id"] for m in s["models"]} >= {"medpsy-4b", "qwen3.5-9b", "qwen3.5-4b"}
        assert s["modelsDir"] == str(tmp_path / "models") and s["chatgpt"]["signedIn"] is False
        req = {"backend": "local", "question": "What is it?", "ctx": {"structures": [{"name": "Pons", "volumeMl": 14}]}}
        st, pv = call(port, "/api/ai/preview", req)
        assert st == 200 and "Pons: 14.0 mL" in pv["preview"]
        st, evs = chat(port, req)
        assert st == 200 and "".join(e.get("t", "") for e in evs) == "An answer" and evs[-1] == {"done": True}
        assert not any("caveat" in e for e in evs)
        evs = chat(port, dict(req, question="What about its size?"))[1]
        assert evs[-2] == {"caveat": ai_chat.CAVEAT} and evs[-1] == {"done": True}   # every backend gets the note
        evs = chat(port, dict(req, question="fail"))[1]
        assert evs[-1] == {"error": "Out of memory.", "fix": "lighter"}   # errors arrive in the stream, with their fix
        assert s["setup"]["step"] == "idle" and s["setup"]["plan"]["id"]
        assert all(m["text"] in pv["preview"] for m in got[0]["messages"]) and got[0]["instructions"] in pv["preview"]
        assert call(port, "/api/ai/chat", dict(req, backend="nope"))[0] == 400
        assert call(port, "/api/ai/chat", dict(req, question=""))[0] == 400
        assert call(port, "/api/ai/models/download", {"id": "../etc"})[0] == 400
        assert call(port, "/api/ai/models/delete", {"id": "../../x"})[0] == 400
        assert call(port, "/api/ai/chat", req, origin="https://evil.example")[0] == 403   # Origin check still applies
        assert no_egress == []
    finally:
        httpd.shutdown()

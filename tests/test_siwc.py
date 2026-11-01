"""Sign in with ChatGPT against a mock authorization server + mock Responses API (no real account or token)."""
import base64
import hashlib
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("cryptography")   # test-only: signs the mock ID tokens (the app checks RS256 with the stdlib)
from cryptography.hazmat.primitives import hashes  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402

from bodyglass import siwc  # noqa: E402

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
ISSUED = "oaiapp_test123"


def b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def jwt(claims, key=KEY):
    h = b64(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    p = b64(json.dumps(claims).encode())
    return f"{h}.{p}." + b64(key.sign(f"{h}.{p}".encode(), padding.PKCS1v15(), hashes.SHA256()))


class Fake:
    def __init__(self):
        self.scope = siwc.SCOPES
        self.calls, self.api_calls, self.nonce, self.challenge, self.redirect = [], [], None, None, None
        self.refresh = "rt-1"
        self.sub = "user-abc"
        self.bad_sig = False


def make_server(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            if self.path == "/.well-known/openid-configuration":
                return self.reply(200, {"issuer": base, "jwks_uri": base + "/jwks", "revocation_endpoint": base + "/revoke"})
            if self.path == "/jwks":
                n = KEY.public_key().public_numbers()
                return self.reply(200, {"keys": [{"kty": "RSA", "kid": "k1", "alg": "RS256",
                                                  "n": b64(n.n.to_bytes(256, "big")), "e": b64(n.e.to_bytes(3, "big"))}]})
            if self.path == "/v1/models":
                fake.api_calls.append(("models", self.headers["Authorization"]))
                return self.reply(200, {"models": [{"slug": "gpt-x", "display_name": "GPT X", "visibility": "list"},
                                                   {"slug": "hidden", "visibility": "hide"}]})
            self.reply(404, {})

        def do_POST(self):
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/v1/responses":
                body = json.loads(raw)
                fake.api_calls.append(("responses", self.headers["Authorization"], body))
                ev = [{"type": "response.created"}, {"type": "response.output_text.delta", "delta": "Hello, "},
                      {"type": "response.output_text.delta", "delta": "world!"}, {"type": "response.completed"}]
                return self.reply(200, "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode(),
                                  "text/event-stream")
            f = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            fake.calls.append((self.path, f))
            if self.path == "/revoke":
                return self.reply(200, b"")
            if f.get("grant_type") == "authorization_code":
                ok = (f["client_id"] == ISSUED and f["code"] == "code-1" and f["redirect_uri"] == fake.redirect
                      and f["resource"] == "https://api.openai.com/v1"
                      and b64(hashlib.sha256(f["code_verifier"].encode()).digest()) == fake.challenge)
                if not ok:
                    return self.reply(400, {"error": "invalid_grant"})
                claims = {"iss": base, "aud": ISSUED, "sub": fake.sub, "email": "test@example.com",
                          "exp": time.time() + 3600, "nonce": fake.nonce}
                tok = jwt(claims, rsa.generate_private_key(public_exponent=65537, key_size=2048) if fake.bad_sig else KEY)
                return self.reply(200, {"access_token": "at-1", "refresh_token": fake.refresh, "id_token": tok,
                                        "token_type": "Bearer", "expires_in": 3600, "scope": fake.scope})
            if f.get("grant_type") == "refresh_token":
                if f["refresh_token"] != fake.refresh or f["client_id"] != ISSUED or "scope" in f:
                    return self.reply(400, {"error": "invalid_grant"})
                fake.refresh = "rt-2"
                return self.reply(200, {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600,
                                        "scope": fake.scope})
            self.reply(404, {})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class MemStore:
    def __init__(self): self.v = None
    def get(self): return self.v
    def set(self, v): self.v = v
    def delete(self): self.v = None


@pytest.fixture
def env(tmp_path):
    fake = Fake()
    srv = make_server(fake)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    opened = []

    def browser(url):   # plays the user's browser: approve, then redirect to the loopback callback
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        opened.append(q)
        fake.nonce, fake.challenge, fake.redirect = q["nonce"], q["code_challenge"], q["redirect_uri"]
        back = {"code": "code-1", "state": q["state"], "scope": fake.scope}
        if q["client_id"] == "dynamic_agent_client":
            back["client_id"] = ISSUED
        target = q["redirect_uri"] + "?" + urllib.parse.urlencode(back)
        threading.Timer(0.05, lambda: urllib.request.urlopen(target).read()).start()

    store = MemStore()
    c = siwc.ChatGPT(str(tmp_path), store=store, auth=base, api=base + "/v1", open_browser=browser, port=0)
    yield c, fake, opened, store, tmp_path
    srv.shutdown()


def sign_in(c):
    c.start_signin()
    for _ in range(200):
        if not c.pending:
            break
        time.sleep(0.02)


def test_first_sign_in_registers_validates_and_keeps_tokens_out_of_files(env):
    c, fake, opened, store, d = env
    sign_in(c)
    st = c.status()
    assert st["signedIn"] and st["sharing"] and st["email"] == "test@example.com" and not c.error, c.error
    q = opened[0]
    assert q["client_id"] == "dynamic_agent_client" and q["agent_name_hint"] == "BodyGlass"
    assert q["ext_agent_host_id"].startswith("urn:uuid:") and q["response_type"] == "code"
    assert q["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert q["resource"] == "https://api.openai.com/v1" and q["code_challenge_method"] == "S256"
    assert urlparse(q["redirect_uri"]).hostname == "127.0.0.1" and q["redirect_uri"].endswith("/auth/callback")
    assert json.loads(store.v)["refresh_token"] == "rt-1"
    for name in os.listdir(d):   # only host id + account mapping on disk, owner-only, no tokens
        text = open(os.path.join(d, name)).read()
        assert "rt-1" not in text and "at-1" not in text
        assert os.stat(os.path.join(d, name)).st_mode & 0o077 == 0
    assert json.load(open(d / "chatgpt-account.json"))["client_id"] == ISSUED


def test_responses_call_streams_and_follows_the_preview_rules(env):
    c, fake, *_ = env
    sign_in(c)
    assert c.models() == [{"slug": "gpt-x", "name": "GPT X"}]
    text = c.respond("gpt-x", "Be educational.", [{"role": "user", "content": "hi"}])
    assert text == "Hello, world!"
    _, auth, body = fake.api_calls[-1]
    assert auth == "Bearer at-1" and body["store"] is False and body["stream"] is True
    assert body["instructions"] == "Be educational." and isinstance(body["input"], list)
    assert not {"temperature", "max_output_tokens", "previous_response_id", "metadata"} & set(body)


def test_no_call_without_the_plan_usage_scope(env):
    c, fake, *_ = env
    fake.scope = "openid profile email offline_access"
    sign_in(c)
    assert c.status()["signedIn"] and not c.status()["sharing"] and "not allowed" in c.error
    with pytest.raises(siwc.ChatGPTError):
        c.respond("gpt-x", "x", [])
    assert fake.api_calls == []


def test_bad_state_and_bad_signature_are_refused(env):
    c, fake, opened, store, _ = env
    c.pending = {"state": "s", "nonce": "n", "verifier": "v", "redirect": "r", "client_id": None}
    with pytest.raises(siwc.ChatGPTError, match="state"):
        c.finish({"state": "other", "code": "x", "client_id": ISSUED})
    c.pending = None
    fake.bad_sig = True
    sign_in(c)
    assert "signature" in (c.error or "") and store.v is None


def test_refresh_rotates_and_reauth_reuses_the_issued_client(env):
    c, fake, opened, store, _ = env
    sign_in(c)
    rec = json.loads(store.v)
    rec["expires_at"] = time.time() - 1
    store.v = json.dumps(rec)
    assert c.access_token() == "at-2"
    assert json.loads(store.v)["refresh_token"] == "rt-2"
    host = opened[0]["ext_agent_host_id"]
    sign_in(c)
    q = opened[1]
    assert q["client_id"] == ISSUED and "agent_name_hint" not in q and q["ext_agent_host_id"] == host
    assert q["id_token_hint"] and q["login_hint"] == "test@example.com"


def test_sign_out_revokes_and_clears_tokens_but_keeps_registration(env):
    c, fake, opened, store, d = env
    sign_in(c)
    assert c.sign_out() is True
    path, form = fake.calls[-1]
    assert path == "/revoke" and form["token"] == "rt-1" and form["token_type_hint"] == "refresh_token"
    assert form["client_id"] == ISSUED
    assert store.v is None and not c.status()["signedIn"]
    assert json.load(open(d / "chatgpt-account.json"))["client_id"] == ISSUED and (d / "chatgpt-host.json").exists()


def test_requests_outside_openai_hosts_are_refused(env):
    c, *_ = env
    with pytest.raises(siwc.ChatGPTError):
        c._http("https://example.com/v1/responses")


def test_a_second_sign_in_attempt_replaces_the_first(env):
    c = env[0]
    c.open_browser = lambda url: None   # the user never finishes the first attempt
    c.signin_wait = 1
    c.start_signin()
    first = c.pending
    c.signin_wait = 30
    c.start_signin()
    second = c.pending
    time.sleep(2.5)   # the first listener times out without touching the new attempt
    assert second is not first and c.pending is second
    c.pending = None

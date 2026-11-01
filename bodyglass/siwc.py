"""Sign in with ChatGPT (open-source/local-app flow) and the Responses API, run from the local Python server.

Follows developers.openai.com/siwc/token-sharing-open-source (links in design/siwc/LINKS.md): authorization code + PKCE,
loopback redirect on 127.0.0.1, dynamic client registration with a persisted ext_agent_host_id, ID-token check
against OpenAI's JWKS, and no request before the granted scopes include chatgpt.tokens.use.direct.
Tokens live in the macOS Keychain (keyring); only the non-secret host id and account mapping are in files.
The browser page never sees a token: it talks only to the local server (CSP unchanged)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

AUTH = "https://auth.openai.com"
API = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
NEED = "chatgpt.tokens.use.direct"
APP_NAME = "BodyGlass"
USAGE_URL = "https://chatgpt.com/settings/usage"
FRIENDLY = {
    "subscription_sharing_usage_limit_exceeded": "Your ChatGPT plan's usage limit for this app was reached. See ChatGPT Settings > Usage.",
    "subscription_sharing_user_not_eligible": "ChatGPT plan usage isn't available for this account or workspace.",
    "subscription_sharing_usage_unavailable": "ChatGPT couldn't check your usage right now. Try again in a minute.",
    "subscription_sharing_user_unavailable": "ChatGPT account information is temporarily unavailable. Try again in a minute.",
    "subscription_sharing_invalid_user": "ChatGPT could not validate this account. Sign out and sign in again.",
}
DEAD_REFRESH = {"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
                "refresh_token_invalidated", "refresh_token_reused"}


class ChatGPTError(Exception):
    def __init__(self, msg, fix=None):   # fix: the panel's one-tap remedy ('signin', 'usage', 'retry')
        super().__init__(msg)
        self.fix = fix


class KeychainStore:
    """One JSON credential record in the macOS Keychain (via keyring; other OSes use keyring's own backend)."""
    SERVICE, USER = "BodyGlass", "chatgpt"

    def get(self):
        import keyring
        return keyring.get_password(self.SERVICE, self.USER)

    def set(self, value):
        import keyring
        keyring.set_password(self.SERVICE, self.USER, value)

    def delete(self):
        import keyring
        try:
            keyring.delete_password(self.SERVICE, self.USER)
        except keyring.errors.PasswordDeleteError:
            pass


def _b64d(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


_SHA256_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def rs256_ok(n, e, signed, sig):
    """RSASSA-PKCS1-v1_5 / SHA-256 signature check (what RS256 means), stdlib only."""
    k = (n.bit_length() + 7) // 8
    if len(sig) != k:
        return False
    em = pow(int.from_bytes(sig, "big"), e, n).to_bytes(k, "big")
    want = b"\x00\x01" + b"\xff" * (k - 3 - len(_SHA256_INFO) - 32) + b"\x00" + _SHA256_INFO + hashlib.sha256(signed).digest()
    return hmac.compare_digest(em, want)


def verify_id_token(token, jwks, issuer, client_id, nonce, now=None):
    try:
        h64, p64, s64 = token.split(".")
        head, claims = json.loads(_b64d(h64)), json.loads(_b64d(p64))
    except ValueError:
        raise ChatGPTError("The ID token is malformed")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == head.get("kid") and k.get("kty") == "RSA"), None)
    if head.get("alg") != "RS256" or not key:
        raise ChatGPTError("The ID token is not signed with a known OpenAI key")
    n, e = int.from_bytes(_b64d(key["n"]), "big"), int.from_bytes(_b64d(key["e"]), "big")
    if not rs256_ok(n, e, f"{h64}.{p64}".encode(), _b64d(s64)):
        raise ChatGPTError("The ID token signature is invalid")
    aud = claims.get("aud")
    if claims.get("iss") != issuer or client_id not in (aud if isinstance(aud, list) else [aud]):
        raise ChatGPTError("The ID token was issued for someone else")
    if claims.get("exp", 0) < (now or time.time()) or claims.get("nonce") != nonce:
        raise ChatGPTError("The ID token has expired or does not match this sign-in")
    return claims


class ChatGPT:
    def __init__(self, data_dir, store=None, auth=AUTH, api=API, open_browser=webbrowser.open, port=1455):
        self.dir, self.store, self.auth, self.api = data_dir, store or KeychainStore(), auth, api
        self.open_browser, self.port = open_browser, port
        self.pending = None
        self.signin_wait = 300   # seconds the loopback listener waits for the browser
        self.error = None
        self.lock = threading.Lock()   # serialise refreshes: the refresh token rotates
        os.makedirs(data_dir, exist_ok=True)

    # -- small files (non-secret) ---------------------------------------
    def _read(self, name):
        try:
            with open(os.path.join(self.dir, name)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _write(self, name, obj):
        p = os.path.join(self.dir, name)
        fd = os.open(p + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
        os.replace(p + ".tmp", p)

    def host_id(self):
        """Chosen once per installation before the first sign-in; opaque (urn:uuid), never user-identifying."""
        h = self._read("chatgpt-host.json").get("ext_agent_host_id")
        if not h:
            h = f"urn:uuid:{uuid.uuid4()}"
            self._write("chatgpt-host.json", {"ext_agent_host_id": h})
        return h

    def creds(self):
        raw = self.store.get()
        return json.loads(raw) if raw else None

    def status(self):
        try:
            c = self.creds()
        except Exception as e:   # e.g. no Keychain backend
            return {"signedIn": False, "sharing": False, "pending": False, "error": f"Keychain unavailable: {e}", "usageUrl": USAGE_URL}
        acct = self._read("chatgpt-account.json")
        scopes = (c or {}).get("scopes", [])
        return {"signedIn": bool(c), "email": acct.get("email") if c else None, "sharing": bool(c) and NEED in scopes,
                "pending": bool(self.pending), "error": self.error, "usageUrl": USAGE_URL}

    # -- HTTP -------------------------------------------------------------
    def _http(self, url, form=None, token=None, body=None):
        if not (url.startswith(self.auth + "/") or url.startswith(self.api + "/")):
            raise ChatGPTError("Refused a request outside OpenAI's sign-in and API hosts")
        headers = {"Accept": "application/json"}
        data = None
        if form is not None:
            data, headers["Content-Type"] = urlencode(form).encode(), "application/x-www-form-urlencoded"
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120)

    def _json(self, url, **kw):
        try:
            with self._http(url, **kw) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"detail": raw.decode(errors="replace")[:300]}

    def _discovery(self):
        st, d = self._json(self.auth + "/.well-known/openid-configuration")
        if st != 200:
            raise ChatGPTError("Could not reach OpenAI sign-in")
        return d

    # -- sign in ------------------------------------------------------------
    def start_signin(self):
        """Opens the system browser; the loopback listener finishes the sign-in in the background."""
        acct, c = self._read("chatgpt-account.json"), self.creds() or {}
        verifier = _b64e(secrets.token_bytes(48))
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        try:
            srv = HTTPServer(("127.0.0.1", self.port), _Callback)
        except OSError:   # 1455 busy: only the port may vary
            srv = HTTPServer(("127.0.0.1", 0), _Callback)
        redirect = f"http://127.0.0.1:{srv.server_address[1]}/auth/callback"
        client_id = acct.get("client_id")
        q = {"client_id": client_id or "dynamic_agent_client", "ext_agent_host_id": self.host_id(),
             "response_type": "code", "redirect_uri": redirect, "scope": SCOPES, "resource": API, "state": state,
             "nonce": nonce, "code_challenge_method": "S256",
             "code_challenge": _b64e(hashlib.sha256(verifier.encode()).digest())}
        if client_id:
            if c.get("id_token") or acct.get("id_token_hint"):
                q["id_token_hint"] = c.get("id_token") or acct.get("id_token_hint")
            if acct.get("email"):
                q["login_hint"] = acct["email"]
        else:
            q["agent_name_hint"] = APP_NAME
        self.pending = {"state": state, "nonce": nonce, "verifier": verifier, "redirect": redirect, "client_id": client_id}
        self.error = None
        srv.result, srv.timeout = None, 1
        threading.Thread(target=self._listen, args=(srv, self.signin_wait), daemon=True).start()
        url = f"{self.auth}/api/accounts/authorize?{urlencode(q)}"
        self.open_browser(url)
        return url

    def _listen(self, srv, wait):
        end, mine = time.time() + wait, self.pending   # a newer attempt replaces self.pending: this one then stops
        try:
            while srv.result is None and time.time() < end and self.pending is mine:
                srv.handle_request()
            if self.pending is not mine:
                return
            if srv.result is None:
                self.error = "Sign-in timed out. Try again."
                return
            self.finish(srv.result)
        except ChatGPTError as e:
            self.error = str(e)
        except Exception as e:  # network etc.: report, keep the app running
            self.error = f"Sign-in failed: {e}"
        finally:
            if self.pending is mine:
                self.pending = None
            srv.server_close()

    def finish(self, params):
        p = self.pending
        if not p or params.get("state") != p["state"]:
            raise ChatGPTError("Sign-in reply did not match this attempt (state); ignored")
        if params.get("error"):
            raise ChatGPTError("Sign-in was cancelled or ChatGPT plan use was declined."
                               if params["error"] == "access_denied" else f"Sign-in failed: {params['error']}")
        cid = p["client_id"]
        if cid:
            if params.get("client_id") and params["client_id"] != cid:
                raise ChatGPTError("Sign-in returned a different app registration; refused")
        else:
            cid = params.get("client_id")
            if not cid or cid == "dynamic_agent_client":
                raise ChatGPTError("Registration did not complete (no client id). Try again.")
            acct = self._read("chatgpt-account.json")
            self._write("chatgpt-account.json", dict(acct, client_id=cid))   # saved before the exchange, per the docs
        st, tok = self._json(self.auth + "/api/accounts/oauth/token", form={
            "grant_type": "authorization_code", "client_id": cid, "code": params.get("code", ""),
            "code_verifier": p["verifier"], "redirect_uri": p["redirect"], "resource": API})
        if st != 200 or not tok.get("id_token"):
            raise ChatGPTError(f"Sign-in failed at the token step ({tok.get('error') or st}). Try again.")
        disc = self._discovery()
        st, jwks = self._json(disc["jwks_uri"])
        claims = verify_id_token(tok["id_token"], jwks, disc.get("issuer", self.auth), cid, p["nonce"])
        acct = self._read("chatgpt-account.json")
        if acct.get("subject") and acct["subject"] != claims["sub"]:
            raise ChatGPTError("That is a different ChatGPT account than the one registered here; refused")
        self._save_tokens(tok)
        self._write("chatgpt-account.json", dict(acct, client_id=cid, subject=claims["sub"], email=claims.get("email"),
                                                 issuer=claims["iss"]))
        if NEED not in (tok.get("scope") or "").split():
            self.error = "Signed in, but ChatGPT plan use was not allowed. Sign in again and allow it to chat."

    def _save_tokens(self, tok, old=None):
        old = old or {}
        self.store.set(json.dumps({
            "id_token": tok.get("id_token") or old.get("id_token"), "access_token": tok["access_token"],
            "refresh_token": tok.get("refresh_token") or old.get("refresh_token"), "token_type": tok.get("token_type", "Bearer"),
            "expires_at": time.time() + int(tok.get("expires_in", 3600)),
            "scopes": (tok.get("scope") or " ".join(old.get("scopes", []))).split(), "saved_at": time.time()}))

    # -- tokens -------------------------------------------------------------
    def access_token(self):
        """A usable access token with ChatGPT plan permission, refreshed near expiry; raises otherwise."""
        with self.lock:
            c = self.creds()
            if not c:
                raise ChatGPTError("You're not signed in to ChatGPT.", "signin")
            if NEED not in c.get("scopes", []):
                raise ChatGPTError("ChatGPT plan use wasn't allowed when you signed in.", "signin")
            if c["expires_at"] - 120 > time.time():
                return c["access_token"]
            cid = self._read("chatgpt-account.json").get("client_id")
            st, tok = self._json(self.auth + "/api/accounts/oauth/token", form={
                "grant_type": "refresh_token", "client_id": cid, "refresh_token": c.get("refresh_token") or "", "resource": API})
            if st != 200:
                code = tok.get("error") if isinstance(tok.get("error"), str) else (tok.get("error") or {}).get("code")
                if code in DEAD_REFRESH:
                    self.store.delete()
                    raise ChatGPTError("Your ChatGPT sign-in has expired.", "signin")
                raise ChatGPTError(f"Couldn't renew the ChatGPT sign-in ({code or st}).", "retry")
            self._save_tokens(tok, old=c)
            c = self.creds()
            if NEED not in c["scopes"]:
                raise ChatGPTError("ChatGPT plan use is no longer allowed for BodyGlass.", "signin")
            return c["access_token"]

    def sign_out(self):
        """Revoke the renewable session, then clear the tokens; the registration and host id are kept."""
        c, acct = self.creds(), self._read("chatgpt-account.json")
        confirmed = True
        if c and c.get("refresh_token"):
            try:
                ep = self._discovery().get("revocation_endpoint")
                st, _ = self._json(ep, form={"token": c["refresh_token"], "token_type_hint": "refresh_token",
                                             "client_id": acct.get("client_id", "")})
                confirmed = st == 200
            except Exception:
                confirmed = False
        if c and c.get("id_token"):
            self._write("chatgpt-account.json", dict(acct, id_token_hint=c["id_token"]))
        self.store.delete()
        return confirmed

    # -- inference ----------------------------------------------------------
    def models(self):
        st, d = self._json(self.api + "/models", token=self.access_token())
        if st != 200:
            raise ChatGPTError(_err(st, d), _fix(st, d))
        return [{"slug": m["slug"], "name": m.get("display_name") or m["slug"]}
                for m in d.get("models", []) if m.get("visibility") == "list"]

    def respond(self, model, instructions, input_items):
        return "".join(self.respond_stream(model, instructions, input_items))

    def respond_stream(self, model, instructions, input_items):
        """POST /v1/responses with store:false, stream:true; yields text deltas; raises unless response.completed arrives."""
        body = {"model": model, "instructions": instructions, "input": input_items, "store": False, "stream": True}
        try:
            r = self._http(self.api + "/responses", token=self.access_token(), body=body)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                d = json.loads(raw)
            except ValueError:
                d = {"detail": raw.decode(errors="replace")[:300]}
            raise ChatGPTError(_err(e.code, d), _fix(e.code, d))
        with r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                t = ev.get("type")
                if t == "response.output_text.delta":
                    yield ev.get("delta", "")
                elif t == "response.completed":
                    return
                elif t in ("response.failed", "error"):
                    err = (ev.get("response") or {}).get("error") or ev.get("error") or ev
                    raise ChatGPTError(FRIENDLY.get(err.get("code"), f"ChatGPT failed: {err.get('code') or err.get('message')}"),
                                       _fix(0, {"error": err}))
                elif t == "response.incomplete":
                    raise ChatGPTError("ChatGPT stopped before finishing the answer.", "retry")
        raise ChatGPTError("The ChatGPT answer was cut off.", "retry")


def _fix(status, d):
    code = (d.get("error") or {}).get("code") if isinstance(d.get("error"), dict) else None
    if code == "subscription_sharing_usage_limit_exceeded":
        return "usage"
    if status == 401 or code in ("subscription_sharing_invalid_user",):
        return "signin"
    return "retry"


def _err(status, d):
    e = d.get("error") if isinstance(d.get("error"), dict) else {}
    if e.get("code") in FRIENDLY:
        return FRIENDLY[e["code"]]
    msg = e.get("message") or d.get("detail") or ""
    return f"ChatGPT request failed ({status}). {msg}".strip()


class _Callback(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path != "/auth/callback":
            self.send_error(404)
            return
        self.server.result = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = b"<!doctype html><title>BodyGlass</title><p style='font:16px system-ui;margin:3em'>" \
               b"Done. You can close this tab and go back to BodyGlass.</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

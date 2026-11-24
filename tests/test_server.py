"""The local server answers only its own page: no DNS rebinding (Host) and no cross-site POSTs (Origin)."""
import http.client
import threading

from bodyglass import server


def req(port, method, path, host=None, origin=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{port}"}
    if origin:
        h["Origin"] = origin
    c.request(method, path, body=b'{"folder": "/nonexistent"}' if method == "POST" else None, headers=h)
    return c.getresponse().status


def test_host_and_origin_checks(tmp_path):
    httpd = server.serve(0, str(tmp_path))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert req(port, "GET", "/api/cases") == 200
        assert req(port, "GET", f"/api/cases", host=f"localhost:{port}") == 200
        assert req(port, "GET", "/api/cases", host="evil.example:80") == 403          # DNS rebinding
        assert req(port, "POST", "/api/open", origin="https://evil.example") == 403   # cross-site POST
        assert req(port, "POST", "/api/open", origin=f"http://127.0.0.1:{port}") == 400   # own page: reaches the handler
    finally:
        httpd.shutdown()


CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self' blob: data:; worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def headers(port, method, path, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=b'{"folder": "/nonexistent"}' if method == "POST" else None,
              headers={"Host": host or f"127.0.0.1:{port}"})
    r = c.getresponse()
    r.read()
    return r.status, r


def test_every_response_refuses_framing_and_sniffing(tmp_path):
    """No other site may frame the viewer (clickjacking) and the full header set survives later edits."""
    httpd = server.serve(0, str(tmp_path))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        for method, path, host, code in (("GET", "/index.html", None, 200),     # static file
                                         ("GET", "/api/cases", None, 200),      # GET API
                                         ("POST", "/api/open", None, 400),      # POST API
                                         ("GET", "/api/cases", "evil.example:80", 403),
                                         ("GET", "/nope.js", None, 404)):
            st, r = headers(port, method, path, host)
            assert st == code, path
            assert r.getheader("Content-Security-Policy") == CSP, (method, path)
            assert r.getheader("X-Content-Type-Options") == "nosniff", (method, path)
            assert r.getheader("Referrer-Policy") == "no-referrer", (method, path)
    finally:
        httpd.shutdown()

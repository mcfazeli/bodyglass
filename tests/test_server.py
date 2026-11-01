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

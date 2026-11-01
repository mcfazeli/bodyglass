"""BodyGlass: python -m bodyglass [DICOM_FOLDER] [--port N] [--no-browser]"""
import argparse
import os
import threading
import webbrowser

from . import server


def main():
    ap = argparse.ArgumentParser(prog="BodyGlass")
    ap.add_argument("folder", nargs="?", help="DICOM folder to open (optional)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cases", default=None, help="where processed cases are cached")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    try:
        httpd = server.serve(a.port, a.cases)
    except OSError as e:
        raise SystemExit(f"Port {a.port} is busy ({e.strerror}). Is BodyGlass already open? "
                         f"Close the other window, or start with: ./BodyGlass.command --port {a.port + 1}")
    root = a.cases or server.default_cases_root()
    if a.folder:
        server.start_processing(os.path.abspath(os.path.expanduser(a.folder)), root)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"BodyGlass running at {url}  (local only; Ctrl+C to quit)")
    print(f"Your saved models are stored in: {root}")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

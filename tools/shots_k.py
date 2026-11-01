"""Screenshots of the viewer (camera, library, mouth, panels, series list, welcome), PUBLIC sample case only, no faces:
skin hidden or seen from behind, the sagittal slice picture (face profile) hidden. Output: screenshots/k/."""
import os, sys, tempfile, threading

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bodyglass import server  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES, CASE, OUT = os.path.join(ROOT, "cases_h"), "4728e9bc0db7", os.path.join(ROOT, "screenshots", "k")
NOFACE = "#insets figure:nth-child(3) { display:none !important; }"
SHOTS = [  # name, cases root, query, extra JS after ready
    ("k1_camera_focus_neck", CASES, f"case={CASE}&op=skin:0,skull:0&focus=vertebra_c4&hint=1&cam=left&tab=layers", None),
    ("k2_add_scan_library", CASES, f"case={CASE}&op=skin:0,skull:0&library=1&cam=backleft", "document.getElementById('btnAddMore').click()"),
    ("k3_mouth_skin_back", CASES, "case=8a5a68916555&cam=back&select=skin&reskin=1&tab=layers", None),
    ("k4_calm_panel_explore", CASES, f"case={CASE}&op=skin:0,skull:0&q=hippo&select=hippocampus_L&planes=axial&cut=1&cam=backleft&tab=explore", None),
    ("k5_series_plain_words", CASES, f"case={CASE}&op=skin:0,skull:0&details=1&tab=scans&cam=backright", None),
    ("k6_welcome_bodyglass", None, "", None),
    ("k6b_command_palette", CASES, f"case={CASE}&op=skin:0,skull:0&palette=c5&cam=backleft", None),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    empty = tempfile.mkdtemp(prefix="bg_empty_cases_")
    servers = {}
    for root in {CASES, empty}:
        h = server.serve(0, root); threading.Thread(target=h.serve_forever, daemon=True).start(); servers[root] = h
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for name, root, q, js in SHOTS:
            if len(sys.argv) > 1 and not any(a in name for a in sys.argv[1:]):
                continue
            port = servers[root or empty].server_address[1]
            pg = b.new_page(viewport={"width": 1440, "height": 900}, bypass_csp=True)
            errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/?shot=1&{q}", wait_until="commit", timeout=120_000)
            if root:
                pg.wait_for_function("window.__ready === true", timeout=300_000)
            else:
                pg.wait_for_selector("#welcome:not(.hidden)", timeout=120_000)
            pg.add_style_tag(content=NOFACE)
            if js:
                pg.evaluate(js)
            pg.wait_for_timeout(2500)
            pg.screenshot(path=os.path.join(OUT, name + ".png"))
            print(name, "errors:", errs)
            pg.close()
        b.close()


if __name__ == "__main__":
    main()

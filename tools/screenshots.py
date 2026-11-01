"""Capture viewer screenshots with headless Chrome + SwiftShader (software WebGL).

usage: screenshots.py BASE_URL name=query [name=query ...]   (or no shots -> default Phase 1 set)
"""
import sys, time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
import os
OUT = os.environ.get("SHOT_DIR", "/workspace/bodyglass/screenshots")
SHOTS = dict(a.split("=", 1) for a in sys.argv[2:]) or {
    "01_peel_xray_skin": "peel=2&cam=3q",
    "02_slices_cutaway": "peel=0&cam=3qhigh&planes=axial,coronal,sagittal&cut=1&pos=axial:28,coronal:-12,sagittal:0",
    "03_labels_deep_structures": "peel=4&cam=3qhigh&select=ventricles&details=1",
    "05_brain_surface": "peel=2&op=skin:0,skull:0&cam=left&select=brain",
}
W, H = 1600, 960
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                          args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                "--ignore-gpu-blocklist", "--enable-webgl"])
    for name, q in SHOTS.items():
        size = {"width": W, "height": H}
        if "&vw=" in q:  # custom viewport for tiles
            vw = int(q.split("&vw=")[1].split("&")[0]); size = {"width": vw, "height": vw}
        pg = b.new_page(viewport=size, device_scale_factor=1, bypass_csp=True)
        pg.on("pageerror", lambda e: print("pageerror:", e))
        t = time.time()
        acts = [kv for kv in q.split("&") if kv.startswith("__")]
        q2 = "&".join(kv for kv in q.split("&") if not kv.startswith("__"))
        pg.goto(BASE + "?shot=1&" + q2)
        pg.wait_for_function("window.__ready === true", timeout=300_000)
        pg.wait_for_timeout(1200)
        for a in acts:  # __drag=c-axial:-60  __wheel=c-axial:240  __hover=c-axial:0.4:0.5
            kind, spec = a[2:].split("=", 1); el, *vals = spec.split(":")
            bb = pg.locator("#" + el).bounding_box()
            cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
            if kind == "drag":
                pg.mouse.move(cx, cy); pg.mouse.down()
                for k in range(1, 21):
                    pg.mouse.move(cx, cy + float(vals[0]) * k / 20); pg.wait_for_timeout(30)
                pg.mouse.move(cx + 1, cy + float(vals[0]))  # keep pointer down: the tooltip shows the slice number
            elif kind == "wheel":
                pg.mouse.move(cx, cy); pg.mouse.wheel(0, float(vals[0]))
            elif kind == "hover":
                pg.mouse.move(bb["x"] + bb["width"] * float(vals[0]), bb["y"] + bb["height"] * float(vals[1]))
            pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/{name}.png")
        pg.close()
        print(name, f"{time.time()-t:.1f}s", flush=True)
    b.close()

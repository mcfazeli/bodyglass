"""Headless UI regressions (one page load shared by the tests):
1. camera: double-click focuses and orbits a part, two-finger (trackpad) swipe pans, 0 fits all;
   every control of the old left panel is still reachable in the new rail / slice bar / menus; ⌘K finds commands and parts;
2. with "Cut away" on, every on/off mix of the 3 slice planes keeps rotate/zoom working and throws no page error
   (bug: Coronal+Sagittal froze the 3D view; a plane hit in the label-occlusion test had no structId). Dev-only: needs playwright in .venv, system Chrome and the PUBLIC sample case (LEMON, CC0) below,
otherwise skipped. Never point it at real scans."""
import itertools, json, os, shutil, threading

import pytest

from bodyglass import server

CASES = os.environ.get("BODYGLASS_UI_CASES", os.path.join(os.path.dirname(__file__), "..", "cases_h"))
CASE = "4728e9bc0db7"   # public sample: OpenNeuro ds000221 sub-010002 (+ ds001919 neck)
CHROME = shutil.which("google-chrome") or "/usr/bin/google-chrome"
sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.skipif(not (os.path.exists(os.path.join(CASES, CASE, "manifest.json")) and os.path.exists(CHROME)),
                                reason="public sample case or Chrome missing")

SET = """async ([on]) => {
  for (const n of ['axial', 'coronal', 'sagittal']) { const c = document.querySelector(`input[data-plane=${n}]`); if (c.checked !== on.includes(n)) c.click(); }
  const cut = document.getElementById('cutaway'); if (!cut.checked) cut.click();
  const want = { axial: 137, coronal: 124, sagittal: 95 };   // target slice numbers (sliders are in mm)
  const go = (r, v) => { r.value = v; r.dispatchEvent(new Event('input', { bubbles: true })); };
  const frames = () => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok)));   // slice numbers update next frame
  const num = (n) => +document.querySelector(`[data-num=${n}]`).textContent;
  for (const n in want) {
    const r = document.querySelector(`input[data-pos=${n}]`), lo = +r.min;
    go(r, lo + 40); await frames(); const sp = 40 / (num(n) - 1);
    go(r, lo + (want[n] - 1) * sp);
  }
}"""


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    os.environ["BODYGLASS_HOME"] = str(tmp_path_factory.mktemp("bg_home"))   # AI models/sign-in data: not the real folder
    httpd = server.serve(0, CASES)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1280, "height": 800}, bypass_csp=True)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:{port}/?shot=1&case={CASE}&op=skin:0,skull:0")
        pg.wait_for_function("window.__ready === true", timeout=300_000)
        pg.base = f"http://127.0.0.1:{port}/"; pg.browser_ = b; pg.httpd = httpd
        yield pg, errs
        b.close()
    httpd.shutdown()


FOCUSED = """() => { const B = window.__bv, T = B.controls.target, tag = document.getElementById('focusTag').textContent;
  const S = Object.values(B.structById).find((x) => tag.includes(x.st.name)); if (!S) return { tag };
  B.centered.updateMatrixWorld(true); const c = new B.THREE.Box3().setFromObject(S.item.mesh).getCenter(new B.THREE.Vector3());
  return { tag, off: c.distanceTo(T) }; }"""
WHERE = """(id) => { const B = window.__bv, S = B.structById[id]; B.centered.updateMatrixWorld(true);
  const c = new B.THREE.Box3().setFromObject(S.item.mesh).getCenter(new B.THREE.Vector3()).project(B.camera);
  return [(c.x + 1) / 2 * innerWidth, (1 - c.y) / 2 * innerHeight]; }"""
DIST = "() => window.__bv.camera.position.distanceTo(window.__bv.controls.target)"


def test_camera_focus_pan_and_fit_all(page):
    pg, errs = page
    x, y = pg.evaluate(WHERE, "spinal_cord")
    pg.mouse.dblclick(x, y); pg.wait_for_timeout(1500)
    f = pg.evaluate(FOCUSED)
    assert "Orbiting around" in f["tag"] and f.get("off", 99) < 5, f      # orbit centre = the clicked part
    t0, d0 = pg.evaluate("window.__bv.controls.target.toArray()"), pg.evaluate(DIST)
    pg.evaluate("document.getElementById('gl').dispatchEvent(new WheelEvent('wheel', { deltaX: 60, deltaY: 30, bubbles: true, cancelable: true }))")
    pg.evaluate("() => { for (let i = 0; i < 90; i++) window.__bv.controls.update(); }")   # damping settles over frames; headless rAF is throttled
    t1, d1 = pg.evaluate("window.__bv.controls.target.toArray()"), pg.evaluate(DIST)
    assert sum((a - b) ** 2 for a, b in zip(t0, t1)) ** 0.5 > 2 and abs(d1 - d0) / d0 < 0.02, (t0, t1, d0, d1)  # pan, no zoom
    pg.keyboard.press("0")                                                 # fit all steps back to show head + neck
    pg.wait_for_function("(d) => window.__bv.camera.position.distanceTo(window.__bv.controls.target) > d",
                         arg=d1 * 1.3, timeout=20_000)                     # the 0.7 s glide is slow under headless rAF
    assert not errs, errs


# every control of the old panel: [selector, how to reveal it in the new layout]
OLD_CONTROLS = [("#peel", "layers"), ("#layers .layer", "layers"), ("#structList", "layers"), ("#lblToggle", "layers"), ("#lobeColors", "layers"),
                ("#scanChips", "scans"), ("#seriesDetails", "scans"), ("#seriesInfo", "scans"),
                ("input[data-plane=axial]", None), ("input[data-pos=sagittal]", None), ("#cutaway", None),
                ("#keepDeep", "slice"), ("#labelOverlay", "slice"), ("#labelOpacity", "slice"), ("#contrast", "slice"),
                ("#mMeasure", "tools"), ("#mExport", "tools"), ("#mExample", "tools"), ("#btnHelp", None), ("#btnTour", None),
                ("#btnAuto", None), ("#btnReset", None), ("#btnAdd", None), ("#btnPerson", None), ("#search", "explore"), ("#aboutKeys", "about")]
REVEAL = """([sel, how]) => { document.body.click();
  if (how === 'slice') document.getElementById('sliceMore').click();
  else if (how === 'tools') document.getElementById('btnTools').click();
  else if (how) { const P = document.getElementById('panel'); P.classList.add('closed'); document.querySelector(`#rail button[data-tab=${how}]`).click(); }
  const el = document.querySelector(sel); return !!el && el.getClientRects().length > 0; }"""


def test_every_old_control_is_still_reachable(page):
    pg, errs = page
    hidden = [sel for sel, how in OLD_CONTROLS if not pg.evaluate(REVEAL, [sel, how])]
    assert not hidden and not errs, (hidden, errs)


def test_palette_finds_commands_and_parts(page):
    pg, errs = page
    pg.keyboard.press("Control+k")
    assert pg.is_visible("#palette")
    pg.keyboard.type("export"); assert "Export for 3D printing" in pg.inner_text("#palList")
    pg.fill("#palIn", "C5"); pg.wait_for_timeout(100)
    assert "C5" in pg.inner_text("#palList").split("\n")[0]                 # a body part comes first
    pg.keyboard.press("Enter"); pg.wait_for_timeout(1500)
    assert pg.is_hidden("#palette") and "C5" in pg.inner_text("#focusTag")   # Enter focuses that part
    assert not errs, errs


def test_cutaway_plane_combos_keep_controls_alive(page):
    pg, errs = page
    errs.clear()
    bad = {}
    for r in range(4):
        for combo in itertools.combinations(["axial", "coronal", "sagittal"], r):
            pg.evaluate(SET, [list(combo)])
            cam0 = pg.evaluate("window.__bv.camera.position.toArray()")
            pg.mouse.move(700, 400); pg.mouse.down()
            for k in range(10):
                pg.mouse.move(700 + 15 * k, 400 + 3 * k); pg.wait_for_timeout(40)
            pg.mouse.up(); pg.mouse.wheel(0, -200); pg.wait_for_timeout(900)   # > the 350 ms label-occlusion throttle
            cam1 = pg.evaluate("window.__bv.camera.position.toArray()")
            moved = sum((a - c) ** 2 for a, c in zip(cam0, cam1)) ** 0.5
            if errs or moved < 1:
                bad["+".join(combo) or "none"] = (round(moved, 1), errs[:1])
            errs.clear()
    assert not bad, bad


def test_example_volunteer_opens_without_slices(page):
    pg0, _ = page
    pg = pg0.browser_.new_page(viewport={"width": 1280, "height": 800}, bypass_csp=True)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(pg0.base + "?shot=1&case=example:e95531164f7c", wait_until="commit", timeout=120_000)
    pg.wait_for_function("window.__ready === true", timeout=300_000)
    assert pg.evaluate("document.body.classList.contains('novol') && Object.keys(window.__bv.layers).length > 2")
    assert "Example volunteer" in pg.inner_text("#btnPerson") and not errs, errs
    pg.close()


class FakeLocal:
    """Stands in for the on-device model (and the one-tap setup) so the UI flow runs in seconds."""
    def __init__(self):
        self.ready, self.got = False, []

    def usable(self, mid): return self.ready and mid == "qwen3.5-4b"

    def stream(self, payload, want=None):
        import time
        self.got.append(payload)
        yield {"model": "Qwen3.5-4B", "note": None, "private": True}
        q = payload["messages"][-1]["text"]
        if "slow" in q:
            for _ in range(200):
                time.sleep(0.05)
                yield {"t": "more "}
        for w in ["The ", "left hippocampus ", "helps you ", "**remember**."]:
            yield {"t": w}


class FakeSetup:
    def __init__(self, loc): self.loc, self.polls = loc, 0
    def start(self, variant="4"): self.polls = 1; return True

    def status(self):
        plan = {"id": "qwen3.5-4b", "name": "Qwen3.5-4B", "vision": True, "runtime": "mlx", "ramGB": 16, "downloadGB": 9.3,
                "diskGB": 2.9, "prebuiltGB": 3.0}
        if not self.polls:
            return {"step": "idle", "plan": plan}
        self.polls += 1
        self.loc.ready = self.polls > 3   # (the server lists models before asking setup: ready shows on the next poll)
        if self.polls > 4:
            return {"step": "ready", "plan": plan}
        return {"step": "download", "plan": plan, "pct": 40, "etaS": 300, "doneGB": 3.7, "totalGB": 9.3}


def test_ask_ai_setup_ask_attach_link_report_and_stop(page):
    pg, errs = page
    errs.clear()
    urls, A = [], pg.httpd.RequestHandlerClass.ai
    loc = FakeLocal()
    A["backends"]["local"], A["setup"] = loc, FakeSetup(loc)
    pg.on("request", lambda r: urls.append(r.url))
    pg.evaluate("Object.keys(localStorage).filter((k) => k.startsWith('bg.ai') || k.startsWith('bg.chats')).forEach((k) => localStorage.removeItem(k))")
    pg.click("#btnChat")                                                  # 1: Ask AI
    pg.wait_for_selector("#aiWelcome:not(.hidden)")
    assert not pg.is_visible("#composer") and "Qwen3.5-4B" in pg.inner_text("#tlDesc")
    pg.click("#btnSetup")                                                 # 2: Set up
    pg.wait_for_function("document.getElementById('tlProg').textContent.includes('min left')", timeout=10_000)
    assert "40%" in pg.inner_text("#btnChat")                            # progress also on the button
    pg.wait_for_selector("#composer:not(.hidden)", timeout=15_000)
    assert pg.is_visible("#suggest .sq") and "Private" in pg.inner_text("#sentLine")
    pg.click("#suggest .sq")                                              # 3: a suggested question -> first answer
    pg.wait_for_selector(".msg.a a.part")
    assert "remember" in pg.inner_text(".msg.a b") and pg.get_attribute("#sendBtn", "title").startswith("Send")
    pg.click(".msg.a a.part")                                             # a part name in the answer focuses it in 3D
    pg.wait_for_selector("#info:not(.hidden)")
    assert "hippocampus" in pg.inner_text("#info").lower()
    chip = pg.inner_text("#chips")
    assert "Left hippocampus" in chip and "mL" in chip                   # the selection became a chip
    pg.click("#attachBtn")
    pg.click("#attachMenu [data-a=axial]")
    pg.wait_for_selector("#chips img")
    pg.click("#chips .chip:has-text('On screen') .rm")                   # removable
    pg.fill("#chatQ", "What is this?")
    pg.keyboard.press("Enter")
    pg.wait_for_function("document.querySelectorAll('.msg.a .tools').length === 2")
    sent = loc.got[-1]
    assert sent["image"].startswith("data:image/png;base64,") and "Left hippocampus" in json.dumps(sent["messages"])
    assert "Shown on my screen" not in json.dumps(sent["messages"]) and not pg.query_selector("#chips img")   # picture went once
    pg.fill("#chatQ", "slow one")
    pg.click("#sendBtn")
    pg.wait_for_function("document.getElementById('sendBtn').classList.contains('stop')")
    pg.wait_for_timeout(400)
    pg.click("#sendBtn")                                                  # Stop
    pg.wait_for_function("document.getElementById('chatLog').textContent.includes('(stopped)')")
    pg.click("#attachBtn")
    pg.click("#attachMenu [data-a=report]")
    pg.fill("#reportText", "Findings: The brainstem and the third ventricle are normal. Mild asymmetry of the hippocampi.")
    pg.click("#reportGo")
    pg.wait_for_function("document.querySelectorAll('.msg.a .tools').length === 4")
    report_links = pg.evaluate("[...document.querySelectorAll('.msg.u')].pop().textContent")
    assert "Brainstem" in report_links or "brainstem" in report_links, report_links
    assert "My report (pasted)" in json.dumps(loc.got[-1]["messages"])
    pg.reload()                                                           # chats are kept on this computer
    pg.wait_for_function("window.__ready === true", timeout=300_000)
    pg.click("#btnChat")
    pg.click("#chatHist")
    assert pg.locator("#histList .hrow").count() == 1
    assert all(u.startswith(pg.base) or u.startswith("data:") or u.startswith("blob:") for u in urls), urls
    pg.click("#chatClose")
    assert not errs, errs


PINCH = """() => { const B = window.__bv, cv = document.getElementById('gl'), T = B.controls.target;
  const dist = () => B.camera.position.distanceTo(T), settle = () => { for (let i = 0; i < 90; i++) B.controls.update(); };
  const at = { clientX: innerWidth / 2, clientY: innerHeight / 2, bubbles: true, cancelable: true };
  const gesture = (type, scale, el) => { const e = new Event(type, at); Object.defineProperty(e, 'scale', { value: scale });
    el.dispatchEvent(e); return e.defaultPrevented; };
  settle(); const d0 = dist();
  const wheel = new WheelEvent('wheel', { ...at, deltaY: 8, ctrlKey: true });   // Chrome/Firefox pinch (fingers together)
  cv.dispatchEvent(wheel); settle(); const d1 = dist();
  const panelWheel = new WheelEvent('wheel', { ...at, deltaY: 8, ctrlKey: true });
  document.getElementById('panel').dispatchEvent(panelWheel); settle(); const d2 = dist();
  const g = [gesture('gesturestart', 1, cv), gesture('gesturechange', 0.85, cv), gesture('gestureend', 0.85, cv)];   // Safari
  settle(); const d3 = dist();
  const scroll = new WheelEvent('wheel', { ...at, deltaY: 40 });   // plain scroll in a panel stays a scroll
  document.getElementById('panel').dispatchEvent(scroll);
  return { d0, d1, d2, d3, wheel: wheel.defaultPrevented, panel: panelWheel.defaultPrevented, g, scroll: scroll.defaultPrevented,
           vv: visualViewport.scale, meta: document.querySelector('meta[name=viewport]').content }; }"""


def test_trackpad_pinch_zooms_the_model_not_the_page(page):
    pg, errs = page
    errs.clear()
    pg.keyboard.press("0")   # a known, unclamped distance
    pg.wait_for_timeout(1500)
    r = pg.evaluate(PINCH)
    assert r["wheel"] and r["d1"] > r["d0"] * 1.02, r              # ctrl-wheel on the canvas zooms the camera
    assert r["panel"] and abs(r["d2"] - r["d1"]) < 1e-6, r         # over a panel: page zoom blocked, camera untouched
    assert all(r["g"]) and r["d3"] > r["d2"] * 1.1, r              # Safari gesture zooms too and is blocked for the page
    assert not r["scroll"], r
    assert r["vv"] == 1 and "user-scalable=no" in r["meta"] and not errs, (r, errs)

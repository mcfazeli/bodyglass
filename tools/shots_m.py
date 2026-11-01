"""Screenshots of Ask AI. PUBLIC sample case only, no faces (sagittal inset hidden,
skin off). Honest stand-ins, labelled in the model names: Mac hardware is mocked (Apple silicon, 16 GB) so the welcome
card shows what a Mac sees; text answers come from llama.cpp with MedPsy-1.7B GGUF, the picture answer from mlx_vlm with
a CANNED stand-in (the real 0.8B picture model needs 12-20+ min on this CPU; it was checked once);
ChatGPT is a mocked client (no account, no network).
Usage: python tools/shots_m.py [--no-image]   Output: screenshots/m/."""
import json, os, sys, tempfile, threading, time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HOME = tempfile.mkdtemp(prefix="bg_ai_home_m_")
os.environ["BODYGLASS_HOME"] = HOME
from bodyglass import ai_chat, ai_models, server, siwc  # noqa: E402

GGUF = "/workspace/aitest/medpsy-1.7b-iq3_xxs.gguf"
QWEN = "/workspace/aitest/qwen35-0.8b-4bit"
CASES, CASE = os.path.join(ROOT, "cases_h"), "4728e9bc0db7"
OUT = os.path.join(ROOT, "screenshots", sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "m")
NOFACE = "#insets figure:nth-child(3) { display:none !important; }"
Q = f"case={CASE}&op=skin:0,skull:0&cam=backleft&tab=explore"
REPORT = ("MRI BRAIN WITHOUT CONTRAST (public example text). FINDINGS: The ventricles are normal in size. "
          "Mild asymmetry of the hippocampi, left slightly smaller than right. The brainstem and cerebellum are "
          "unremarkable. No mass effect. IMPRESSION: No acute abnormality. Mild hippocampal asymmetry, of uncertain significance.")


class PictureStandIn:
    """Text questions go to the real local backend; a picture question gets the real model choice and swap note
    (Local.pick) but a canned answer, labelled as such."""
    def __init__(self, real): self.real = real
    def usable(self, mid): return self.real.usable(mid)

    def stream(self, payload, want=None):
        if not payload["image"]:
            yield from self.real.stream(payload, want)
            return
        mid, note = self.real.pick(True, want)
        yield {"model": self.real.models.model(mid)["name"] + " (canned demo answer: no model ran)", "note": note, "private": True}
        for w in ("This is an **axial slice**: a view from above, as if the head were cut horizontally. ",
                  "Brighter areas are white matter, the grey rim is the cortex, and the dark shapes in the middle are "
                  "the ventricles, fluid spaces.\n\nEducational only: your doctor can explain what your own scan shows."):
            yield {"t": w}


class MockChatGPT:   # screenshot only: looks signed in, never touches the network
    expired = False

    def status(self):
        return {"signedIn": True, "email": "you@example.com", "sharing": True, "pending": False, "error": None,
                "usageUrl": "https://chatgpt.com/settings/usage"}

    def models(self):
        return [{"slug": "demo", "name": "GPT (demo)"}]

    def respond_stream(self, *a):
        raise siwc.ChatGPTError("Your ChatGPT sign-in has expired.", "signin")
        yield


def main():
    mdir = os.path.join(HOME, "models")
    os.makedirs(mdir, exist_ok=True)
    man = json.load(open(ai_models.BUNDLED))
    for m in man["models"]:
        m["name"] = {"medpsy-4b-gguf": "MedPsy-1.7B (stand-in)", "qwen3.5-4b": "Qwen3.5-4B"}.get(m["id"], m["name"])
    json.dump(man, open(os.path.join(mdir, "ai_models.json"), "w"))
    h = server.serve(0, CASES)
    A = h.RequestHandlerClass.ai
    A["setup"] = ai_models.Setup(A["models"], lambda k: True, silicon=lambda: True, ram=lambda: 16)   # a 16 GB M-series Mac
    A["backends"]["local"] = PictureStandIn(A["backends"]["local"])
    threading.Thread(target=h.serve_forever, daemon=True).start()
    port = h.server_address[1]
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1440, "height": 900}, bypass_csp=True)  # only so the script can poll; app CSP unchanged
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))

        def shot(name):
            pg.wait_for_timeout(1200)
            pg.screenshot(path=os.path.join(OUT, name + ".png"))
            print(name, "errors:", errs, flush=True)

        def answered(n):
            pg.wait_for_function(f"document.querySelectorAll('.msg.a .tools, .msg.a.err').length >= {n}", timeout=600_000)

        pg.goto(f"http://127.0.0.1:{port}/?shot=1&{Q}", wait_until="commit", timeout=120_000)
        pg.wait_for_function("window.__ready === true", timeout=300_000)
        pg.add_style_tag(content=NOFACE)
        pg.evaluate("localStorage.clear()")
        pg.click("#btnChat")
        pg.wait_for_selector("#aiWelcome:not(.hidden)")
        shot("m1_first_open")
        mg = A["models"]   # Set up is tapped; its download is faked (real status/ETA code, no network, no 9 GB)
        A["setup"].start = lambda variant="4": mg.state.update({"qwen3.5-4b": {
            "status": "downloading", "variant": "4", "done": 3.9e9, "total": 9.3e9, "t0": time.time() - 120, "done0": 0,
            "msg": "Downloading"}}) or True
        pg.click("#btnSetup")
        pg.wait_for_function("document.getElementById('tlProg').textContent.includes('Downloading')", timeout=10_000)
        shot("m2_setup_progress")
        mg.state["qwen3.5-4b"] = {"status": "ready", "path": QWEN, "variant": "prebuilt"}
        mg.state["medpsy-4b-gguf"] = {"status": "ready", "path": GGUF, "variant": "4"}
        pg.wait_for_selector("#composer:not(.hidden)", timeout=10_000)
        pg.evaluate("localStorage.setItem('bg.aiModel', 'local:medpsy-4b-gguf')")   # the fast text stand-in
        pg.goto(f"http://127.0.0.1:{port}/?shot=1&{Q}&select=hippocampus_L", wait_until="commit")
        pg.wait_for_function("window.__ready === true", timeout=300_000)
        pg.add_style_tag(content=NOFACE)
        pg.click("#infoAsk")
        pg.wait_for_selector("#suggest .sq")
        shot("m3_ready_with_suggestions")
        pg.click("#suggest .sq")
        answered(1)
        shot("m4_first_answer_part_links")
        if "--no-image" not in sys.argv:
            pg.click("#chatNew")   # short history: the CPU stand-in reads pictures very slowly
            pg.click("#attachBtn")
            pg.click("#attachMenu [data-a=axial]")
            pg.wait_for_selector("#chips img")
            pg.fill("#chatQ", "What am I looking at in this picture?")
            shot("m5a_picture_attached")
            pg.keyboard.press("Enter")
            answered(1)
            shot("m5_picture_answer_auto_swap")
        n = pg.evaluate("document.querySelectorAll('.msg.a .tools, .msg.a.err').length")
        pg.click("#chatNew")
        pg.click("#attachBtn")
        pg.click("#attachMenu [data-a=report]")
        pg.fill("#reportText", REPORT)
        pg.click("#reportGo")
        pg.wait_for_timeout(1500)
        shot("m6a_report_highlighted")
        answered(1)
        shot("m6_report_explained")
        cg = MockChatGPT()
        A["chatgpt"] = cg
        A["backends"]["chatgpt"].client = cg
        pg.click("#modelBtn")
        pg.wait_for_selector("#modelMenu .mi")
        shot("m7_model_switcher")
        pg.click("#modelMenu .mi:has-text('ChatGPT')")
        pg.wait_for_selector("#cgConsent:not(.hidden)")
        shot("m8_chatgpt_consent")
        pg.click("#cgOk")
        pg.fill("#chatQ", "What is the brainstem?")
        pg.keyboard.press("Enter")
        pg.wait_for_selector(".msg.a.err")
        shot("m9_error_with_fix")
        print("n before report", n)
        b.close()
    h.shutdown()


if __name__ == "__main__":
    main()

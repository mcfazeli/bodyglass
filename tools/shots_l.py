"""Screenshots of the AI chat, PUBLIC sample case only, no faces (sagittal slice hidden,
skin off). The local answer is real (llama.cpp on this Linux box, MedPsy-1.7B GGUF as a labelled stand-in for
MedPsy-4B); the ChatGPT preview uses a mocked signed-in state (no real account). Output: screenshots/l/."""
import json, os, sys, tempfile, threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HOME = tempfile.mkdtemp(prefix="bg_ai_home_")
os.environ["BODYGLASS_HOME"] = HOME
from bodyglass import ai_models, server  # noqa: E402

GGUF = os.environ.get("BODYGLASS_TEST_GGUF", "/workspace/aitest/medpsy-1.7b-iq3_xxs.gguf")
CASES, CASE, OUT = os.path.join(ROOT, "cases_h"), "4728e9bc0db7", os.path.join(ROOT, "screenshots", "l")
NOFACE = "#insets figure:nth-child(3) { display:none !important; }"
Q = f"case={CASE}&op=skin:0,skull:0&select=hippocampus_L&cam=backleft&tab=explore"


class MockChatGPT:   # screenshot only: looks signed in, never touches the network
    def status(self):
        return {"signedIn": True, "email": "you@example.com", "sharing": True, "pending": False, "error": None,
                "usageUrl": "https://chatgpt.com/settings/usage"}

    def models(self):
        return [{"slug": "demo", "name": "ChatGPT model (demo)"}]


def setup_models():
    os.makedirs(os.path.join(HOME, "models"), exist_ok=True)
    man = json.load(open(ai_models.BUNDLED))
    for m in man["models"]:
        if m["id"] == "medpsy-4b-gguf":
            m["name"] = "MedPsy-1.7B GGUF (Linux test stand-in)"
    json.dump(man, open(os.path.join(HOME, "models", "ai_models.json"), "w"))
    json.dump({"medpsy-4b-gguf": {"status": "ready", "path": GGUF, "variant": "4"},
               "qwen3.5-4b": {"status": "paused", "variant": "prebuilt", "total": 3_000_000_000, "done": 1_050_000_000,
                              "msg": "Paused"}}, open(os.path.join(HOME, "models", "state.json"), "w"))


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_models()
    h = server.serve(0, CASES)
    threading.Thread(target=h.serve_forever, daemon=True).start()
    port = h.server_address[1]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1440, "height": 900}, bypass_csp=True)  # only so the test can poll; the app CSP is unchanged
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))

        def load():
            pg.goto(f"http://127.0.0.1:{port}/?shot=1&{Q}", wait_until="commit", timeout=120_000)
            pg.wait_for_function("window.__ready === true", timeout=300_000)
            pg.add_style_tag(content=NOFACE)

        def shot(name):
            pg.wait_for_timeout(1500)
            pg.screenshot(path=os.path.join(OUT, name + ".png"))
            print(name, "errors:", errs)

        load()
        pg.evaluate("localStorage.clear()")
        pg.click("#btnChat")
        shot("l1_chat_off")
        pg.click("#chatOn")
        pg.click("#chatModels")
        pg.wait_for_selector(".mcard")
        shot("l2_ai_models_page")
        pg.click("#aimClose")
        h.RequestHandlerClass.ai["chatgpt"] = MockChatGPT()
        h.RequestHandlerClass.ai["backends"]["chatgpt"].client = h.RequestHandlerClass.ai["chatgpt"]
        pg.click("#aiBackend button[data-b=chatgpt]")
        pg.wait_for_selector("#cgModel option", state="attached")
        pg.check("#shSel")
        pg.check("#shFind")
        pg.fill("#findText", "Example report line: hippocampal volumes within normal limits.")
        pg.fill("#chatQ", "What does this part do, and what does 'within normal limits' mean?")
        pg.click("#chatPrev")
        pg.wait_for_selector("#chatPreview:not(.hidden)")
        shot("l3_send_preview_chatgpt")
        pg.click("#chatBack")
        pg.click("#aiBackend button[data-b=local]")
        pg.uncheck("#shFind")
        pg.fill("#chatQ", "In two sentences: what does this part do?")
        pg.click("#chatPrev")
        pg.click("#chatSend")
        pg.wait_for_function("document.querySelectorAll('#chatLog .msg').length >= 2 && !document.querySelector('#chatLog').textContent.includes('Thinking on this Mac')", timeout=600_000)
        shot("l4_local_answer_sample")
        b.close()
    h.shutdown()


if __name__ == "__main__":
    main()

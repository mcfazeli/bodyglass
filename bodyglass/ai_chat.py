"""AI chat: what gets sent (only what the user ticked) and the two pluggable backends.

A backend is any object with answer(payload) -> {"text", "thinking"}; 'local' (on this Mac, offline) and
'chatgpt' (Sign in with ChatGPT) exist now. The browser never talks to a model: everything goes through
the local server, so the page's CSP stays 'self'-only."""
from __future__ import annotations

import base64
import re
import gc
import os
import tempfile
import threading

INSTRUCTIONS = (
    "You are the educational assistant inside BodyGlass, an app that shows a 3D model built from the user's own MRI. "
    "Explain anatomy, and what measurements mean in general, in plain, friendly words. You may give published typical "
    "ranges in general terms, saying they vary between people and between methods. You are not a doctor: never say "
    "whether the user's own scan, size, measurement or structure is normal, abnormal, healthy, typical for them or "
    "nothing to worry about, and never diagnose. If asked, say that only their doctor can judge that. "
    "Volumes in BodyGlass are rough computer estimates. Name body parts plainly (for example 'left hippocampus'). "
    "Keep answers short: 2 to 5 sentences or a few bullets, unless asked for more detail.")
CAVEAT = "BodyGlass can't judge whether your scan is normal. Ask your doctor."
# Cheap output check: reassurance or a normal/abnormal verdict. shortcut: a phrase list, not a classifier.
VERDICT = re.compile(r"\b(within (the |a )?(typical|normal|healthy|expected) range|nothing to worry|not something to worry|"
                     r"no (cause|need) for (concern|worry)|looks? (healthy|normal)|appears? (healthy|normal)|"
                     r"(is|are|seems?|likely) (completely |perfectly |probably |likely )?(normal|healthy|abnormal)\b|"
                     r"normal (finding|structure|size|scan|result)s?)", re.I)


def needs_caveat(text):
    """True when an answer (thinking removed) sounds like a verdict on the user's scan."""
    return bool(VERDICT.search(split_thinking(text)[0]))


MAX_TEXT = 4000
MAX_REPORT = 12000
MAX_IMAGE = 4_000_000   # data-URL characters, about 3 MB of PNG


def _num(v):
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def build_payload(ctx, question, history=(), image=None):
    """The exact request content, built only from ticked items; the preview shows this same object."""
    ctx = ctx if isinstance(ctx, dict) else {}
    parts = []
    st = [s for s in (ctx.get("structures") or [])[:200] if isinstance(s, dict) and str(s.get("name", "")).strip()]
    if st:
        parts.append("Structures:\n" + "\n".join(
            f"- {str(s['name'])[:80]}" + (f": {_num(s.get('volumeMl'))} mL" if _num(s.get("volumeMl")) is not None else "")
            for s in st))
    ms = [str(m)[:200] for m in (ctx.get("measurements") or [])[:50] if str(m).strip()]
    if ms:
        parts.append("Measurements:\n" + "\n".join(f"- {m}" for m in ms))
    vw = [str(v)[:80] for v in (ctx.get("view") or [])[:30] if str(v).strip()]
    if vw:
        parts.append("Shown on my screen: " + ", ".join(vw))
    if str(ctx.get("findings") or "").strip():
        parts.append("My report (pasted):\n" + str(ctx["findings"]).strip()[:MAX_REPORT])
    q = str(question or "").strip()[:MAX_TEXT]
    if not q:
        raise ValueError("Type a question first")
    text = ("What I chose to share from my scan:\n\n" + "\n\n".join(parts) + "\n\n" if parts else "") + "My question: " + q
    hist = [{"role": h["role"], "text": str(h.get("text", ""))[:MAX_TEXT]} for h in list(history or [])[-12:]
            if isinstance(h, dict) and h.get("role") in ("user", "assistant")]
    if image is not None and not (isinstance(image, str) and image.startswith("data:image/png;base64,") and len(image) < MAX_IMAGE):
        raise ValueError("The attached picture is not a PNG or is too large")
    return {"instructions": INSTRUCTIONS, "messages": hist + [{"role": "user", "text": text, "image": bool(image)}],
            "image": image}


def preview(payload):
    """Human-readable, complete rendering of build_payload's result (nothing else is sent)."""
    out = ["[Instructions]", payload["instructions"]]
    for m in payload["messages"]:
        out += ["", f"[{'You' if m['role'] == 'user' else 'AI'}]", m["text"]]
    if payload["image"]:
        out += ["", "[Attached image] 1 slice picture (PNG, shown above)"]
    return "\n".join(out)


def split_thinking(text):
    """MedPsy is a thinking model: keep its reasoning apart from the answer."""
    if "</think>" in text:
        think, ans = text.split("</think>", 1)
        return ans.strip(), think.replace("<think>", "").strip()
    return text.strip(), ""


def mlx_ok():
    """MLX runtime importable (Apple silicon in practice; Linux only with mlx[cpu], used by the tests)."""
    try:
        import mlx_lm  # noqa: F401
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def llama_ok():
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


class ChatError(Exception):
    """An error the panel shows in one line, with a one-tap fix ('setup', 'image_model', 'lighter', 'signin', 'retry', 'usage')."""
    def __init__(self, msg, fix=None):
        super().__init__(msg)
        self.fix = fix


# Preference when the user has not picked a model: the image-capable Qwen first (it also chats), MedPsy is opt-in.
LOCAL_ORDER = ("qwen3.5-9b", "qwen3.5-4b", "medpsy-4b", "medpsy-4b-gguf")


class Local:
    """On-device models, streamed. One model in memory at a time (two would need ~13 GB)."""

    def __init__(self, models):
        self.models = models
        self.loaded = None   # (model id, runtime, objects)
        self.lock = threading.Lock()

    def usable(self, mid):
        m = self.models.model(mid)
        return bool(self.models.installed(mid)) and (llama_ok() if m["runtime"] == "llama_cpp" else mlx_ok())

    def pick(self, image, want=None):
        """(model id, note). A text-only choice is swapped for an installed image model for a message with a picture."""
        ready = [mid for mid in LOCAL_ORDER if self.usable(mid)]
        if not ready:
            raise ChatError("The private model isn't set up yet.", "setup")
        mid = want if want in ready else ready[0]
        if image and not self.models.model(mid).get("vision"):
            seeing = [x for x in ready if self.models.model(x).get("vision")]
            if not seeing:
                raise ChatError(f"{self.models.model(mid)['name']} reads text only. Add the picture-reading model to ask about images.",
                                "image_model")
            note = f"Answered by {self.models.model(seeing[0])['name']}: {self.models.model(mid)['name']} can't read pictures."
            return seeing[0], note
        return mid, None

    def _load(self, mid):
        if self.loaded and self.loaded[0] == mid:
            return self.loaded[1], self.loaded[2]
        self.loaded = None   # free the previous model before loading the next
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        path, runtime = self.models.installed(mid), self.models.model(mid)["runtime"]
        if runtime == "mlx_lm":
            from mlx_lm import load
            objs = load(path)
        elif runtime == "mlx_vlm":
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            objs = load(path) + (load_config(path),)
        else:
            from llama_cpp import Llama
            objs = (Llama(model_path=path, n_ctx=8192, verbose=False),)
        self.loaded = (mid, runtime, objs)
        return runtime, objs

    def stream(self, payload, want=None):
        """Yields {"model", "note"} first, then {"t": text}; stops as soon as the caller stops iterating (Stop button)."""
        with self.lock:
            mid, note = self.pick(payload["image"], want)
            yield {"model": self.models.model(mid)["name"], "modelId": mid, "note": note, "private": True}
            try:
                runtime, objs = self._load(mid)
                msgs = [{"role": "system", "content": payload["instructions"]}] + \
                       [{"role": m["role"], "content": m["text"]} for m in payload["messages"]]
                if runtime == "mlx_lm":
                    from mlx_lm import stream_generate
                    model, tok = objs
                    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
                    for r in stream_generate(model, tok, prompt, max_tokens=2048):
                        yield {"t": r.text}
                elif runtime == "mlx_vlm":
                    yield from self._vlm(objs, msgs, payload["image"])
                else:
                    for c in objs[0].create_chat_completion(messages=msgs, max_tokens=2048, stream=True):
                        t = c["choices"][0]["delta"].get("content")
                        if t:
                            yield {"t": t}
            except MemoryError:
                self.loaded = None
                raise ChatError("This Mac ran out of memory for that model.", "lighter")
            except RuntimeError as e:   # MLX reports allocation failures as RuntimeError
                if "memory" in str(e).lower():
                    self.loaded = None
                    raise ChatError("This Mac ran out of memory for that model.", "lighter")
                raise

    def _vlm(self, objs, msgs, image):
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor, config = objs
        files = []
        try:
            if image:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                    fh.write(base64.b64decode(image.split(",", 1)[1]))
                files.append(fh.name)
            prompt = apply_chat_template(processor, config, msgs, num_images=len(files))
            for r in stream_generate(model, processor, prompt, files or None, max_tokens=1500):
                yield {"t": r.text}
        finally:
            for f in files:
                os.remove(f)


class Cloud:
    """Sign in with ChatGPT backend; sends exactly build_payload's content via the Responses API."""
    def __init__(self, client):
        self.client = client

    def stream(self, payload, model=None):
        from . import siwc
        items = []
        for m in payload["messages"]:
            if m["role"] == "assistant":
                items.append({"role": "assistant", "content": m["text"]})
            else:
                content = [{"type": "input_text", "text": m["text"]}]
                if m.get("image") and payload["image"]:
                    content.append({"type": "input_image", "image_url": payload["image"]})
                items.append({"role": "user", "content": content})
        try:
            if not self.client.status().get("signedIn"):
                raise ChatError("You're not signed in to ChatGPT.", "signin")
            models = self.client.models()
            slug = model if model in [x["slug"] for x in models] else (models[0]["slug"] if models else None)
            if not slug:
                raise ChatError("Your ChatGPT account has no models available for apps.", None)
            yield {"model": next(x["name"] for x in models if x["slug"] == slug), "modelId": "chatgpt:" + slug,
                   "note": None, "private": False}
            for t in self.client.respond_stream(slug, payload["instructions"], items):
                yield {"t": t}
        except siwc.ChatGPTError as e:
            raise ChatError(str(e), e.fix)
        except OSError:
            raise ChatError("Can't reach ChatGPT. Check your internet connection.", "retry")

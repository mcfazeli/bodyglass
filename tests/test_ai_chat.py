"""AI chat: only chosen context is sent, exact preview, streaming MLX (mocked) and ChatGPT (fake), llama.cpp end to end, no egress."""
import json
import os
import socket
import sys
import types
import urllib.request

import pytest

from bodyglass import ai_chat, ai_models

GGUF = os.environ.get("BODYGLASS_TEST_GGUF", "/workspace/aitest/medpsy-1.7b-iq3_xxs.gguf")
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture
def no_egress(monkeypatch):
    """Socket guard: any connection or DNS lookup that is not loopback is recorded and refused."""
    tried = []
    local = ("127.0.0.1", "::1", "localhost")
    real_connect, real_cx, real_gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def check(sock, addr):
        if sock.family == socket.AF_UNIX or (isinstance(addr, tuple) and addr[0] in local):
            return True
        tried.append(addr)
        return False

    def connect(self, addr):
        if not check(self, addr):
            raise OSError("outbound connection blocked by test")
        return real_connect(self, addr)

    def connect_ex(self, addr):
        return real_cx(self, addr) if check(self, addr) else 111

    def gai(host, *a, **kw):
        if host not in local and host is not None:
            tried.append(("dns", host))
            raise socket.gaierror("DNS blocked by test")
        return real_gai(host, *a, **kw)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return tried


def test_only_ticked_items_are_sent_and_the_preview_is_complete():
    p = ai_chat.build_payload({}, "What is the hippocampus?")
    assert p["messages"] == [{"role": "user", "text": "My question: What is the hippocampus?", "image": False}]
    ctx = {"structures": [{"name": "Left hippocampus", "volumeMl": 3.912}], "measurements": ["Distance 41.2 mm"],
           "findings": "Small cyst noted.", "unexpected": "patient name", "caseId": "abc"}
    p = ai_chat.build_payload(ctx, "Is this size normal?", history=[{"role": "user", "text": "hi"},
                                                                    {"role": "assistant", "text": "hello"},
                                                                    {"role": "system", "text": "evil"}])
    sent = json.dumps(p)
    assert "patient name" not in sent and "abc" not in sent and "evil" not in sent
    assert "Left hippocampus: 3.91 mL" in sent and "Distance 41.2 mm" in sent and "Small cyst noted." in sent
    pv = ai_chat.preview(p)
    assert p["instructions"] in pv and all(m["text"] in pv for m in p["messages"]) and p["image"] is None
    with pytest.raises(ValueError):
        ai_chat.build_payload({}, "  ")
    with pytest.raises(ValueError):
        ai_chat.build_payload({}, "q", image="data:text/html;base64,PHNjcmlwdD4=")
    assert "[Attached image]" in ai_chat.preview(ai_chat.build_payload({}, "q", image=PNG))


def test_thinking_is_kept_apart():
    assert ai_chat.split_thinking("<think>\nhmm\n</think>\n\nAnswer.") == ("Answer.", "hmm")
    assert ai_chat.split_thinking("Plain") == ("Plain", "")


class Installed:
    """Model files present for these ids; names, runtimes and vision flags from the shipped manifest."""
    def __init__(self, ids):
        self.ids, self.man = ids, {m["id"]: m for m in json.load(open(ai_models.BUNDLED))["models"]}
    def installed(self, mid): return f"/models/{mid}" if mid in self.ids else None
    def model(self, mid): return self.man[mid]


def run(gen):
    evs = list(gen)
    return evs[0], "".join(e.get("t", "") for e in evs[1:])


def mock_mlx(monkeypatch, log, seen):
    lm = types.ModuleType("mlx_lm")

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt):
            log.append(("template", [m["role"] for m in msgs], add_generation_prompt))
            return [1, 2, 3]
    lm.load = lambda path: (log.append(("load_lm", path)) or ("LM", Tok()))
    lm.stream_generate = lambda model, tok, prompt, max_tokens: iter([types.SimpleNamespace(text=t) for t in ("<think>t</think>", "Text ", "answer")])
    vlm = types.ModuleType("mlx_vlm")
    vlm.load = lambda path: (log.append(("load_vlm", path)) or ("VLM", "PROC"))

    def vstream(model, proc, prompt, image, max_tokens):
        if image:
            seen.update(exists=os.path.exists(image[0]), png=open(image[0], "rb").read(4), path=image[0])
        yield types.SimpleNamespace(text="Image " if image else "Qwen ")
        yield types.SimpleNamespace(text="answer")
    vlm.stream_generate = vstream
    vu, vp = types.ModuleType("mlx_vlm.utils"), types.ModuleType("mlx_vlm.prompt_utils")
    vu.load_config = lambda path: {"cfg": path}
    vp.apply_chat_template = lambda proc, cfg, msgs, num_images: (log.append(("vtemplate", num_images)) or "P")
    for k, v in {"mlx_lm": lm, "mlx_vlm": vlm, "mlx_vlm.utils": vu, "mlx_vlm.prompt_utils": vp}.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(ai_chat, "mlx_ok", lambda: True)


def test_local_streams_and_swaps_to_a_picture_model_with_mocked_mlx(monkeypatch):
    log, seen = [], {}
    mock_mlx(monkeypatch, log, seen)
    loc = ai_chat.Local(Installed({"medpsy-4b", "qwen3.5-4b"}))
    meta, text = run(loc.stream(ai_chat.build_payload({}, "q")))
    assert meta["modelId"] == "qwen3.5-4b" and meta["private"] and text == "Qwen answer"   # the default sees pictures
    meta, text = run(loc.stream(ai_chat.build_payload({}, "q"), "medpsy-4b"))
    assert meta["modelId"] == "medpsy-4b" and text == "<think>t</think>Text answer"
    assert ("template", ["system", "user"], True) in log
    meta, text = run(loc.stream(ai_chat.build_payload({}, "what is this?", image=PNG), "medpsy-4b"))
    assert meta["modelId"] == "qwen3.5-4b" and "can't read pictures" in meta["note"] and text == "Image answer"
    assert seen["exists"] and seen["png"] == b"\x89PNG" and not os.path.exists(seen["path"])
    assert ("vtemplate", 1) in log and [e[0] for e in log].count("load_vlm") == 2   # one model in memory at a time
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Local(Installed({"medpsy-4b"})).stream(ai_chat.build_payload({}, "q", image=PNG)))
    assert e.value.fix == "image_model"
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Local(Installed(set())).stream(ai_chat.build_payload({}, "q")))
    assert e.value.fix == "setup"


def test_local_out_of_memory_offers_the_lighter_model(monkeypatch):
    mock_mlx(monkeypatch, [], {})
    sys.modules["mlx_vlm"].load = lambda path: (_ for _ in ()).throw(RuntimeError("[metal] Insufficient Memory"))
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Local(Installed({"qwen3.5-9b"})).stream(ai_chat.build_payload({}, "q")))
    assert e.value.fix == "lighter"


def test_cloud_backend_streams_the_same_payload_and_maps_errors():
    from bodyglass import siwc
    sent = {}

    class Client:
        signed = True
        def status(self): return {"signedIn": self.signed}
        def models(self): return [{"slug": "gpt-x", "name": "GPT-X"}]
        def respond_stream(self, model, instructions, items):
            sent.update(model=model, instructions=instructions, items=items)
            yield "o"
            yield "k"
    c = Client()
    p = ai_chat.build_payload({"findings": "note"}, "q", history=[{"role": "assistant", "text": "earlier"}], image=PNG)
    meta, text = run(ai_chat.Cloud(c).stream(p, "nope"))
    assert text == "ok" and meta == {"model": "GPT-X", "modelId": "chatgpt:gpt-x", "note": None, "private": False}
    assert sent["instructions"] == p["instructions"]
    assert sent["items"][0] == {"role": "assistant", "content": "earlier"}
    assert sent["items"][1]["content"] == [{"type": "input_text", "text": p["messages"][-1]["text"]},
                                           {"type": "input_image", "image_url": PNG}]
    c.signed = False
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Cloud(c).stream(p))
    assert e.value.fix == "signin"
    c.signed = True
    c.respond_stream = lambda *a: (_ for _ in ()).throw(siwc.ChatGPTError("You've hit your ChatGPT usage limit.", "usage"))
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Cloud(c).stream(p))
    assert e.value.fix == "usage"
    c.respond_stream = lambda *a: (_ for _ in ()).throw(OSError("offline"))
    with pytest.raises(ai_chat.ChatError) as e:
        run(ai_chat.Cloud(c).stream(p))
    assert e.value.fix == "retry" and "internet" in str(e.value)


def test_socket_guard_really_blocks(no_egress):
    with pytest.raises(OSError):
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    assert no_egress


@pytest.mark.skipif(not (os.path.exists(GGUF) and ai_chat.llama_ok()), reason="tiny public GGUF or llama-cpp-python missing")
def test_local_llama_answers_without_any_outbound_connection(tmp_path, no_egress):
    (tmp_path / "state.json").write_text(json.dumps({"medpsy-4b-gguf": {"status": "ready", "path": GGUF}}))
    loc = ai_chat.Local(ai_models.Manager(str(tmp_path)))
    meta, text = run(loc.stream(ai_chat.build_payload({"structures": [{"name": "Left hippocampus", "volumeMl": 3.9}]},
                                                      "In one sentence, what does this part do?")))
    assert meta["modelId"] == "medpsy-4b-gguf" and len(ai_chat.split_thinking(text)[0]) > 10
    assert no_egress == []


def test_verdicts_on_the_scan_get_a_caveat_and_the_prompt_forbids_them():
    for t in ["3.6 mL is within typical range for adults.", "It's likely showing a normal structure.",
              "This is not something to worry about.", "Your hippocampus looks healthy.", "<think>x</think>It is normal."]:
        assert ai_chat.needs_caveat(t), t
    for t in ["The hippocampus helps form memories.", "Sizes vary a lot between people; only your doctor can judge yours.",
              "<think>is it normal?</think>Ask your doctor."]:
        assert not ai_chat.needs_caveat(t), t
    assert "only their doctor can judge" in ai_chat.INSTRUCTIONS and "2 to 5 sentences" in ai_chat.INSTRUCTIONS

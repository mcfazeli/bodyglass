"""Maintainer tool: write bodyglass/ai_models.json with pinned commits and the Hub's LFS SHA256 per file.

Run by hand when bumping a model; the app itself only reads the bundled file (or re-pins on a user tap)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bodyglass.ai_models import pin_repo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bodyglass", "ai_models.json")
LIC_MEDPSY = "Apache-2.0. Note: trained partly on data with non-commercial terms; fine for personal use."
LIC_QWEN = "Apache-2.0."

MODELS = [
    dict(id="qwen3.5-4b", name="Qwen3.5-4B", purpose="Private assistant that reads text and pictures. Right for Macs with 16 GB of memory or less.", default=True,
         repo="Qwen/Qwen3.5-4B", runtime="mlx_vlm", bits=[4], ramGB={"4": 4.5}, diskGB={"4": 3.1},
         license=LIC_QWEN, prebuilt="mlx-community/Qwen3.5-4B-4bit", vision=True),
    dict(id="qwen3.5-9b", name="Qwen3.5-9B", purpose="Private assistant that reads text and pictures. Sharper answers; for Macs with 24 GB of memory or more.",
         repo="Qwen/Qwen3.5-9B", runtime="mlx_vlm", bits=[4], ramGB={"4": 8.5}, diskGB={"4": 6.0},
         license=LIC_QWEN, prebuilt="mlx-community/Qwen3.5-9B-4bit", vision=True),
    dict(id="medpsy-4b", name="MedPsy-4B", purpose="Medical-tuned, text only. Thinks before it answers, so it is slower.", vision=False, medical=True,
         repo="qvac/MedPsy-4B", runtime="mlx_lm", bits=[4, 8], ramGB={"4": 3.5, "8": 5.5}, diskGB={"4": 2.3, "8": 4.3},
         license=LIC_MEDPSY),
    dict(id="medpsy-4b-gguf", name="MedPsy-4B (GGUF fallback)", purpose="Text-only fallback for Macs without Apple silicon (Intel).", vision=False, medical=True,
         repo="qvac/MedPsy-4B-GGUF", runtime="llama_cpp", bits=[4], ramGB={"4": 3.8}, diskGB={"4": 2.7},
         license=LIC_MEDPSY, only=["medpsy-4b-q4_k_m-imat.gguf", "LICENSE"]),
]

if __name__ == "__main__":
    out = []
    for m in MODELS:
        m = dict(m, **pin_repo(m["repo"], only=m.pop("only", None)))
        if m.get("prebuilt"):
            m["prebuilt"] = pin_repo(m["prebuilt"])
        out.append(m)
        print(m["id"], m["revision"], sum(f["size"] for f in m["files"]) / 1e9, "GB")
    with open(OUT, "w") as fh:
        json.dump({"version": 1, "models": out}, fh, indent=1)

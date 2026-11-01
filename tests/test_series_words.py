"""Every reason dicom_scan can give for skipping or down-ranking a series has a plain-language sentence in the UI."""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_exclusion_and_penalty_has_a_plain_sentence():
    src = open(os.path.join(ROOT, "bodyglass", "dicom_scan.py")).read()
    body = src[src.index("def score_series"):src.index("def pick_t1")]
    labels = set(re.findall(r'excluded = (?:s\.excluded or )?"([^"]+)"', body))
    labels |= set(re.findall(r'\), "([^"]+)"\),', body))                      # the 'hard' exclusion table
    labels |= {re.split(r" \(\{", m)[0] if "{" in m else m
               for m in re.findall(r'add\(-\d+, f?"([^"]+)"\)', body)}       # penalties ("thick slices ({..})" -> "thick slices")
    assert len(labels) >= 12, labels
    app = open(os.path.join(ROOT, "bodyglass", "web", "app.js")).read()
    plain = app[app.index("const PLAIN = {"):app.index("};", app.index("const PLAIN = {"))]
    keys = set(re.findall(r"^\s*'([^']+)':", plain, re.M))
    assert not labels - keys, f"no plain sentence for: {labels - keys}"

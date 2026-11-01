"""Measuring maths in app.js: slice-picture pixel <-> RAS mm must round-trip on all three planes."""
import json
import os
import re
import shutil
import subprocess

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bodyglass", "web", "app.js")


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_inset_ras_round_trip():
    src = open(APP).read()
    fns = "\n".join(re.search(r"function %s\(.*?\n}\n" % n, src, re.S).group(0) for n in ("insetToRas", "rasToInset"))
    js = """
class V { constructor(x, y, z) { this.c = [x, y, z]; } getComponent(a) { return this.c[a]; } }
const THREE = { Vector3: V };
const dims = [120, 140, 100], origin = [-60.5, -80, -40], sp = 1.0, k = { axial: 37, coronal: 81, sagittal: 12 };
const sliceIdx = (n) => k[n];
%s
const out = [];
for (const n of ['axial', 'coronal', 'sagittal']) for (const [px, py] of [[0, 0], [17, 63], [99, 99]]) {
  const p = insetToRas(n, px, py), q = rasToInset(n, p);
  out.push([n, px, py, q.c, q.r, q.d, p.c]);
}
console.log(JSON.stringify(out));""" % fns
    rows = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    for n, px, py, c, r, d, p in rows:
        assert (c, r, d) == (px, py, 0), (n, px, py, c, r, d)
    # axial pixel (0,0) is the patient's right-anterior corner (radiological view, RAS: x right is +)
    ax = [p for n, px, py, *_r, p in rows if n == "axial" and px == 0 and py == 0][0]
    assert ax == [-60.5 + 119, -80 + 139, -40 + 37]

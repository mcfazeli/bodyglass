"""Camera focus maths in app.js: fitDist puts a sphere of radius r exactly inside the narrower field of view."""
import json, math, os, re, shutil, subprocess

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bodyglass", "web", "app.js")


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_fit_distance_uses_the_narrower_field_of_view():
    fn = re.search(r"function fitDist\(.*?\n", open(APP).read()).group(0)
    js = fn + "console.log(JSON.stringify([fitDist(50, 32, 1.6, 1), fitDist(50, 32, 0.5, 1), fitDist(20, 32, 1.6)]));"
    wide, tall, small = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    v = math.radians(16)
    assert abs(wide - 50 / math.sin(v)) < 1e-6                    # landscape window: vertical FOV limits
    h = math.atan(math.tan(v) * 0.5)
    assert abs(tall - 50 / math.sin(h)) < 1e-6 and tall > wide     # portrait window: horizontal FOV limits, farther away
    assert abs(small - 1.2 * 20 / math.sin(v)) < 1e-6              # default 20 % margin

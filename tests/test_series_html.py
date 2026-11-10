"""Synthetic DICOM metadata must remain text in series panels and notifications.

Execute the real rendering functions in Node, then parse their generated HTML.
No browser, scans, network, or scientific dependencies are needed.
"""
import json
import pathlib
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "bodyglass/web/app.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
HOSTILE = '<a href="https://example.invalid">fake link</a><img src=x onerror="bad()"> & &#34; \''
UID = '2.25." autofocus onfocus="bad()"><a href="https://example.invalid">uid</a>'


class Fragment(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def render(manifest, button_uid=None):
    source = APP.read_text()
    helper = re.search(r"^const escH = .*", source, re.M).group()
    series = source[source.index("const PLAIN = {"):source.index("async function useFor3D")]
    add_scan = source[source.index("async function addScan(cid)"):source.index("$('#btnAdd').onclick")]
    script = helper + "\n" + """
const boxes = {};
const $ = id => boxes[id] ||= {innerHTML: '', classList: {add() {}}};
const buttons = BUTTONS;
const rebuilt = [];
const useFor3D = uid => rebuilt.push(uid);
const document = {querySelectorAll: () => buttons};
const sequences = () => [];
const manifest = INPUT;
let notices = [], neckPolls = 0;
const toast = text => { notices.push(text); return {remove() {}}; };
const fetch = async () => ({json: async () => ({folder: true, route: 'neck', neck: manifest.neck.series.description})});
const pollNeck = () => {neckPolls++;};
""".replace("INPUT", json.dumps(manifest)).replace("BUTTONS", json.dumps(
        [] if button_uid is None else [{"dataset": {"uid": button_uid}}])) + series + "\n" + add_scan + """
buildSeriesUI();
buttons.forEach(b => b.onclick());
addScan('synthetic').then(() => console.log(JSON.stringify({boxes, notices, neckPolls, rebuilt})));
"""
    return json.loads(subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout)


def fixture(text=HOSTILE, uid=UID):
    chosen = dict(uid="chosen", description=text, nFiles=120, voxel=[1, 1, 1],
                  acquisition="3D", reasons=["+120 " + text], score=120)
    alternative = dict(uid=uid, description=text, nFiles=80, voxel=[1, 1, 2],
                       acquisition="2D", reasons=["-20 thick slices"], score=80)
    excluded = dict(uid="excluded", description=text, nFiles=5, reasons=[],
                    score=-1000, excluded=text)
    return dict(series=dict(chosen=chosen, all=[chosen, alternative, excluded]),
                neck=dict(series=dict(description=text), status="ok"),
                timing=dict(total=3), brainExtraction=dict(method="synthstrip"),
                volumesMl=dict(brain_tissue=1000), phase2=dict(timing=dict(total=10)))


def assert_safe(html):
    parsed = Fragment(html)
    assert not any(tag in {"a", "img", "script", "iframe", "svg"} for tag, _ in parsed.tags)
    assert not any(key.startswith("on") or key == "autofocus"
                   for _, attrs in parsed.tags for key in attrs)
    return parsed


@pytest.mark.parametrize("panel", ["#seriesInfo", "#seriesPlain", "#seriesTech"])
def test_series_descriptions_and_reasons_are_literal_text(panel):
    result = render(fixture())
    parsed = assert_safe(result["boxes"][panel]["innerHTML"])
    assert HOSTILE in "".join(parsed.text)


def test_series_uid_stays_one_attribute_and_round_trips_for_rebuild():
    result = render(fixture())
    parsed = assert_safe(result["boxes"]["#seriesPlain"]["innerHTML"])
    buttons = [attrs for tag, attrs in parsed.tags if tag == "button"]
    assert buttons == [{"class": "link use3d", "data-uid": UID}]
    assert "Use for 3D instead" in "".join(parsed.text)
    # Feed the parsed attribute back as the browser dataset, then execute the real handler.
    assert render(fixture(), buttons[0]["data-uid"])["rebuilt"] == [UID]


@pytest.mark.parametrize("field", ["nFiles", "voxel", "acquisition", "total", "score"])
def test_technical_fields_cannot_create_markup(field):
    data = fixture("ordinary")
    if field == "total":
        data["timing"]["total"] = HOSTILE
    elif field == "voxel":
        data["series"]["chosen"][field] = [HOSTILE, 1, 2]
    else:
        data["series"]["chosen"][field] = HOSTILE
    parsed = assert_safe(render(data)["boxes"]["#seriesTech"]["innerHTML"])
    if field != "score":  # nonnumeric scores retain the existing 'excluded' fallback
        assert HOSTILE in "".join(parsed.text)


def test_neck_add_notification_is_literal_text_and_starts_polling():
    result = render(fixture())
    parsed = assert_safe(result["notices"][-1])
    assert "".join(parsed.text) == f"Adding “{HOSTILE}” as the neck exam…"
    assert result["neckPolls"] == 1


def test_normal_text_formatting_fallbacks_and_rebuild_affordance_are_preserved():
    data = fixture('T1 & T2 "reference"', "2.25.123")
    result = render(data)
    info = Fragment(result["boxes"]["#seriesInfo"]["innerHTML"])
    assert "".join(info.text) == '3D model built from T1 & T2 "reference"3 series found in this exam · everything ran on this computer'
    assert any(tag == "b" for tag, _ in info.tags)
    plain = Fragment(result["boxes"]["#seriesPlain"]["innerHTML"])
    assert any(attrs.get("data-uid") == "2.25.123" for _, attrs in plain.tags)
    tech = "".join(Fragment(result["boxes"]["#seriesTech"]["innerHTML"]).text)
    assert "120 images · 1×1×1 mm · 3D · built in 3 s" in tech
    assert "SynthStrip (AI, on this computer)" in tech
    assert "FastSurfer (AI, on this computer)" in tech
    for series in data["series"]["all"]:
        series["description"] = ""
    data["neck"]["series"]["description"] = ""
    result = render(data)
    assert "the T1 scan" in result["boxes"]["#seriesInfo"]["innerHTML"]
    assert "(no name)" in result["boxes"]["#seriesPlain"]["innerHTML"]
    assert "neck exam" in result["boxes"]["#seriesPlain"]["innerHTML"]
    assert "(no description)" in result["boxes"]["#seriesTech"]["innerHTML"]

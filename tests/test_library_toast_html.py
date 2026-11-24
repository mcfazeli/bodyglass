"""DICOM text and error messages stay text in the My body library and in notifications.

Runs the real app.js functions in Node with small stubs, then parses the HTML they produce.
Synthetic data only; no browser, scans or network.
"""
import json
import re
import subprocess

import pydicom
import pytest
from pydicom.encaps import encapsulate
from pydicom.uid import UID

from bodyglass import server
from test_multiframe import write_enhanced
from test_series_html import APP, HOSTILE, Fragment, assert_safe, pytestmark  # noqa: F401  (node skip)

STUBS = """
var boxes = {}, notices = [], cards = [], handle = {addEventListener() {}}, history = {replaceState() {}};
var $ = id => boxes[id] ||= {innerHTML: '', textContent: '', classList: {add() {}, remove() {}, toggle() {}},
                             append: el => cards.push(el.innerHTML)};
var document = {createElement: () => ({innerHTML: '', querySelector: () => handle})};
var toast = text => { notices.push(text); return {remove() {}}; };
var liveStatus = () => {}, loadCase = async () => {}, setLoading = () => {}, showWelcome = () => {};
var hideWelcome = () => {}, showSteps = () => {}, hasStructs = () => true;
var manifest = null, currentCase = 'c1', firstBuild = true, neckAvail = false, caseVersion = 0;
var fetch = async url => ({json: async () => (REPLY)[url]});
"""


def run(reply, functions, call):
    """Run the named top-level app.js functions and a call with stubs; return what reached the page."""
    source = APP.read_text()
    code = re.search(r"^const escH = .*", source, re.M).group() + STUBS.replace("REPLY", json.dumps(reply))
    for const in ("fmtDate", "caseGuard", "ACTIVE", "jobOf", "newestBuild"):
        code += "\n" + re.search(rf"^const {const} = .*", source, re.M).group()
    for name in functions:
        start = source.index(f"\nasync function {name}(")
        code += source[start:source.index("\n}\n", start) + 3]
    code += f"\n{call}.then(() => console.log(JSON.stringify({{notices, cards}})));"
    done = subprocess.run(["node", "-e", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


POLLS = ["pollNeck", "pollPhase2", "pollProcessing"]


def test_library_card_shows_exam_kind_description_and_name_as_text():
    case = dict(caseId="c1", createdAt="2026-11-16T12:00:00", description=HOSTILE, name=HOSTILE, structures=True,
                exams=[dict(kind=HOSTILE, description=HOSTILE, date="20261116"),
                       dict(kind="Neck MRI", description=HOSTILE, ok=False)])
    (card,) = run({"api/cases": [case], "api/state": {"casesRoot": "", "mac": False}}, ["openLibrary"], "openLibrary()")["cards"]
    parsed = assert_safe(card)
    assert "".join(parsed.text).count(HOSTILE) == 3   # head kind and description, neck description
    assert [a["value"] for t, a in parsed.tags if t == "input"] == [HOSTILE]   # the name round-trips intact
    assert "Head MRI" not in card and "Neck MRI" in card


def test_library_card_keeps_ordinary_text_and_buttons():
    case = dict(caseId="c1", createdAt=None, description="T1 MPRAGE", name="Me", structures=False,
                exams=[dict(kind="Head MRI", description="T1 MPRAGE & more", date=None)])
    (card,) = run({"api/cases": [case], "api/state": {"casesRoot": "", "mac": False}}, ["openLibrary"], "openLibrary()")["cards"]
    parsed = Fragment(card)
    assert "Head MRI" in parsed.text and " T1 MPRAGE & more " in parsed.text
    assert [t for t, _ in parsed.tags].count("button") == 2   # Close (open case) and Delete


@pytest.mark.filterwarnings("ignore:.*Invalid value for VR UI")
def test_crafted_transfer_syntax_uid_in_the_no_usable_scan_message_is_text(tmp_path):
    uid = '1.2.<a href="//example.invalid">x</a>&amp;<img src=x onerror="bad()">'
    p = tmp_path / "IM0001"
    write_enhanced(p)
    ds = pydicom.dcmread(p)
    ds.file_meta.TransferSyntaxUID = UID(uid)
    ds.PixelData = encapsulate([b"\x00" * 8] * int(ds.NumberOfFrames))
    ds["PixelData"].VR = "OB"
    ds.save_as(p, enforce_file_format=True)
    reply, code = server.add_scan(None, str(tmp_path), str(tmp_path / "cases"))
    assert code == 400 and uid in reply["error"]
    # Shown on Add scan, and by the processing poll when a rebuild fails with the same message.
    state = {"jobs": {"c1": {"phase1": dict(status="error", error=reply["error"])}}}
    for call in ["addScan('c1')", "pollProcessing('c1')"]:
        (notice,) = run({"api/add": reply, "api/state": state}, ["addScan", *POLLS], call)["notices"][1 if call.startswith("add") else 0:]
        assert "".join(assert_safe(notice).text) == reply["error"]


@pytest.mark.parametrize("job, call", [
    ({"neck": dict(status="error", msg=HOSTILE, error=HOSTILE)}, "pollNeck('c1')"),
    ({"phase2": dict(status="error", msg=HOSTILE)}, "pollPhase2()"),
    ({"phase1": dict(status="error", error=HOSTILE)}, "pollProcessing('c1')"),
])
def test_background_job_errors_are_text(job, call):
    (notice,) = run({"api/state": {"jobs": {"c1": job}}}, POLLS, call)["notices"]
    assert HOSTILE in "".join(assert_safe(notice).text)

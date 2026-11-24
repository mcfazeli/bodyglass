"""The viewer's progress polls read only the case they belong to, and stop when another case is opened.

Runs the real app.js poll functions in Node with stubs (see test_library_toast_html). Synthetic job states only.
"""
from test_library_toast_html import POLLS, run
from test_series_html import pytestmark  # noqa: F401  (node skip)

# Records what reached the page and runs each scheduled poll by hand, so a test can switch cases in between.
CAPTURE = """var timers = [], shown = [];
liveStatus = (key, text) => shown.push([key, text || null]);
setLoading = (on, msg) => shown.push(['loading', msg]);
loadCase = async (cid) => shown.push(['load', cid]);
globalThis.setTimeout = (fn) => timers.push(fn);
const next = async () => { await timers.shift()(); for (let i = 0; i < 5; i++) await Promise.resolve(); };
const report = () => notices.push(...shown.map((x) => x.join(': ')));
"""


def shown(jobs, script):
    return run({"api/state": {"jobs": jobs}}, POLLS, "(async () => {" + CAPTURE + script + "; report(); })()")["notices"]


RUNNING = lambda pct: {"status": "running", "pct": pct, "msg": "working"}


def test_detailed_structures_show_only_the_open_case():
    jobs = {"A": {"phase2": {"status": "done"}}, "B": {"phase2": RUNNING(40)}}
    out = shown(jobs, "currentCase = 'A'; hasStructs = () => false; await pollPhase2()")
    assert out == ["✓ Detailed structures ready", "p2: ", "load: A"]   # B's 40 % never shows on A


def test_switching_case_stops_the_old_poll_and_clears_its_progress():
    jobs = {"A": {"phase2": RUNNING(40)}, "B": {}}
    out = shown(jobs, "currentCase = 'A'; await pollPhase2(); currentCase = 'B'; caseVersion++; await next()")
    assert out == ["p2: Adding detailed structures on this computer… 40%", "p2: "]   # cleared, not rescheduled


def test_queued_work_says_it_is_waiting():
    jobs = {"A": {"phase1": RUNNING(50)}, "B": {"phase1": {"status": "queued", "pct": 0, "msg": "Waiting"}, "phase2": {"status": "queued"}}}
    out = shown(jobs, "currentCase = 'B'; await pollProcessing('B'); await pollPhase2('B')")
    assert out == ["p1: ", "loading: Waiting for the other scan to finish", "p2: Detailed structures: waiting for other work to finish…"]
    out = shown(jobs, "currentCase = 'A'; manifest = {}; await pollProcessing('B')")   # a model is open: it stays usable
    assert out == ["p1: Next 3D model: waiting for the other scan to finish…"]


def test_neck_progress_of_another_case_is_not_shown():
    jobs = {"A": {"neck": RUNNING(30)}}
    assert shown(jobs, "currentCase = 'B'; await pollNeck('A')") == ["neck: "]
    assert shown(jobs, "currentCase = 'A'; await pollNeck('A')") == ["neck: Neck exam: working… 30%"]


def test_start_follows_the_newest_build_and_an_interrupted_one_is_reported():
    jobs = {"A": {"phase1": dict(RUNNING(10), id=1, caseId="A")}, "B": {"phase1": dict(RUNNING(70), id=7, caseId="B", msg="newest")}}
    assert shown(jobs, "await pollProcessing()") == ["p1: ", "loading: newest"]
    jobs = {"A": {"phase2": {"status": "interrupted", "msg": "Stopped when BodyGlass was closed. Add the scan again to finish it."}}}
    assert shown(jobs, "currentCase = 'A'; hasStructs = () => false; await pollPhase2()") == [
        "Stopped when BodyGlass was closed. Add the scan again to finish it.", "p2: "]   # toast, then progress cleared

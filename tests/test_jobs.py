"""Background jobs: bounded heavy work, progress bound to its own case, deletion that cannot race a writer,
and truthful state after a restart. Synthetic blocked workers and fake case folders only; no scans or models."""
import json
import os
import threading
import time

import pytest

from bodyglass import jobs as J


def case(root, cid):
    os.makedirs(os.path.join(root, cid), exist_ok=True)
    with open(os.path.join(root, cid, "manifest.json"), "w") as fh:
        json.dump({"caseId": cid}, fh)


def wait(cond, timeout=10):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, "timed out"
        time.sleep(0.01)


class Gate:
    """A synthetic worker that blocks until released and records how many run at once."""
    def __init__(self):
        self.lock, self.now, self.peak, self.ran = threading.Lock(), 0, 0, []

    def job(self, name, fail=False):
        ev = threading.Event()

        def fn(up):
            with self.lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
                self.ran.append(name)
            try:
                up(10, f"{name} working")
                assert ev.wait(10)
                if fail:
                    raise RuntimeError(f"{name} failed")
            finally:
                with self.lock:
                    self.now -= 1
        return fn, ev


def status(jobs, cid, kind):
    return jobs.snapshot().get(cid, {}).get(kind, {}).get("status")


@pytest.mark.parametrize("limit", [1, 2])
def test_heavy_limit_is_never_exceeded_and_queued_work_advances_after_success_and_failure(tmp_path, limit):
    root = str(tmp_path)
    jobs, g, evs = J.Jobs(root, {"heavy": limit}), Gate(), []
    for i, cid in enumerate(["aaa", "bbb", "ccc", "ddd"]):
        fn, ev = g.job(cid, fail=i == 0)          # the first one fails: its slot must still be released
        jobs.submit(cid, "phase1", fn, "heavy", new_case=True)
        evs.append(ev)
    wait(lambda: len(g.ran) == limit)
    time.sleep(0.2)
    assert g.peak == limit and status(jobs, "ddd", "phase1") == "queued"
    for ev in evs:
        ev.set()
    wait(lambda: all(status(jobs, c, "phase1") not in J.ACTIVE for c in ["aaa", "bbb", "ccc", "ddd"]))
    assert g.peak == limit and sorted(g.ran[:limit]) == ["aaa", "bbb", "ccc", "ddd"][:limit]   # first come, first served
    assert [status(jobs, c, "phase1") for c in ["aaa", "bbb", "ccc", "ddd"]] == ["error", "done", "done", "done"]
    assert jobs.snapshot()["aaa"]["phase1"]["error"] == "aaa failed"


def test_progress_belongs_to_its_own_case_and_job(tmp_path):
    root = str(tmp_path)
    case(root, "aaa"); case(root, "bbb")
    jobs, ups, go = J.Jobs(root), {}, threading.Event()

    def grab(cid):
        def fn(up):
            ups[cid] = up
            assert go.wait(10)
        return fn
    jobs.submit("aaa", "phase2", grab("aaa"), "heavy")
    jobs.submit("bbb", "sequences", grab("bbb"), "align")
    wait(lambda: len(ups) == 2)
    for i in range(20):                      # interleaved callbacks from two cases' workers
        ups["aaa"](i, f"a{i}")
        ups["bbb"](100 - i, f"b{i}")
    snap = jobs.snapshot()
    assert (snap["aaa"]["phase2"]["pct"], snap["aaa"]["phase2"]["msg"]) == (19, "a19")
    assert (snap["bbb"]["sequences"]["pct"], snap["bbb"]["sequences"]["msg"]) == (81, "b19")
    assert set(snap["aaa"]) == {"phase2"} and set(snap["bbb"]) == {"sequences"}   # no stage leaks across cases
    go.set()
    wait(lambda: status(jobs, "aaa", "phase2") == "done" and status(jobs, "bbb", "sequences") == "done")
    ups["aaa"](5, "late callback")                       # after its job ended: ignored
    assert jobs.snapshot()["aaa"]["phase2"]["pct"] == 100 and jobs.snapshot()["aaa"]["phase2"]["msg"] != "late callback"


def test_a_stage_cannot_be_queued_twice_for_one_case(tmp_path):
    case(str(tmp_path), "aaa")
    jobs, g = J.Jobs(str(tmp_path)), Gate()
    fn, ev = g.job("aaa")
    jobs.submit("aaa", "neck", fn, "heavy")
    with pytest.raises(J.Busy):
        jobs.submit("aaa", "neck", fn, "heavy")
    ev.set()
    wait(lambda: status(jobs, "aaa", "neck") == "done")


def test_deletion_is_refused_while_a_writer_is_queued_or_running(tmp_path):
    root = str(tmp_path)
    case(root, "aaa"); case(root, "bbb")
    jobs, g = J.Jobs(root), Gate()
    run_a, ev_a = g.job("aaa")
    run_b, ev_b = g.job("bbb")

    def writes(fn, cid):   # the worker writes into its case folder once released
        def w(up):
            fn(up)
            with open(os.path.join(root, cid, "out.txt"), "w") as fh:
                fh.write("x")
        return w
    jobs.submit("aaa", "phase2", writes(run_a, "aaa"), "heavy")
    jobs.submit("bbb", "neck", writes(run_b, "bbb"), "heavy")      # queued behind aaa
    wait(lambda: status(jobs, "aaa", "phase2") == "running")
    assert status(jobs, "bbb", "neck") == "queued"
    for cid in ("aaa", "bbb"):                                        # running, queued
        with pytest.raises(J.Busy):
            jobs.delete(cid, lambda: pytest.fail("deleted a case with pending work"))
    ev_a.set(); ev_b.set()
    wait(lambda: status(jobs, "bbb", "neck") == "done")
    import shutil
    jobs.delete("bbb", lambda: shutil.rmtree(os.path.join(root, "bbb")))   # idle now: allowed
    assert not os.path.exists(os.path.join(root, "bbb")) and "bbb" not in jobs.snapshot()


def test_check_to_delete_boundary_admits_no_writer(tmp_path):
    root = str(tmp_path)
    case(root, "aaa")
    jobs, inside, release = J.Jobs(root), threading.Event(), threading.Event()
    import shutil

    def slow_rmtree():   # deletion paused between its check and the removal
        inside.set()
        assert release.wait(10)
        shutil.rmtree(os.path.join(root, "aaa"))
    t = threading.Thread(target=jobs.delete, args=("aaa", slow_rmtree))
    t.start()
    assert inside.wait(10)
    for kind, lane in (("phase2", "heavy"), ("sequences", "align"), ("neck", "heavy"), ("upgrade", "align")):
        with pytest.raises(J.Busy):
            jobs.submit("aaa", kind, lambda up: open(os.path.join(root, "aaa", "late"), "w").close(), lane)
    with pytest.raises(J.Busy):
        with jobs.writing("aaa"):
            pytest.fail("a rename ran during deletion")
    release.set(); t.join(10)
    with pytest.raises(J.Busy):   # after the deletion: no manifest, so follow-up work is not admitted either
        jobs.submit("aaa", "phase2", lambda up: os.makedirs(os.path.join(root, "aaa")), "heavy")
    time.sleep(0.2)
    assert not os.path.exists(os.path.join(root, "aaa"))


def test_a_writer_holding_the_case_blocks_deletion(tmp_path):
    case(str(tmp_path), "aaa")
    jobs = J.Jobs(str(tmp_path))
    with jobs.writing("aaa"):
        with pytest.raises(J.Busy):
            jobs.delete("aaa", lambda: pytest.fail("deleted during a rename"))
    jobs.delete("aaa", lambda: None)


def test_restart_reports_cut_off_work_as_interrupted_and_a_retry_clears_it(tmp_path):
    root = str(tmp_path)
    case(root, "aaa")
    open(os.path.join(root, "aaa", ".job-phase2"), "w").close()    # left behind by a closed app
    jobs = J.Jobs(root)
    P = jobs.snapshot()["aaa"]["phase2"]
    assert P["status"] == "interrupted" and "Add the scan again" in P["msg"]
    assert not jobs.busy("aaa")                                     # not reported as running; deletable
    jobs.submit("aaa", "phase2", lambda up: None, "heavy")          # retry
    wait(lambda: status(jobs, "aaa", "phase2") == "done")
    assert not os.path.exists(os.path.join(root, "aaa", ".job-phase2"))
    assert "aaa" not in J.Jobs(root).snapshot()                     # next start: nothing stale


def test_running_work_leaves_a_marker_until_it_ends(tmp_path):
    case(str(tmp_path), "aaa")
    jobs, g = J.Jobs(str(tmp_path)), Gate()
    fn, ev = g.job("aaa")
    jobs.submit("aaa", "neck", fn, "heavy")
    marker = os.path.join(str(tmp_path), "aaa", ".job-neck")
    wait(lambda: os.path.exists(marker))
    assert J.Jobs(str(tmp_path)).snapshot()["aaa"]["neck"]["status"] == "interrupted"   # as a crash right now would leave it
    ev.set()
    wait(lambda: not os.path.exists(marker))

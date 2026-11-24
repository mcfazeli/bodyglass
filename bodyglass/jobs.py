"""Background work on saved cases: one registry for every job, bounded heavy work, and deletion that cannot race
a writer.

Each job belongs to one case and one stage (phase1, phase2, sequences, neck, upgrade) from the moment it is
queued; its progress callback updates only that job's record. Jobs run on lanes with a fixed number of workers,
first come first served: "heavy" (SynthStrip, FastSurfer, TotalSegmentator; one at a time so two scans never
need their memory at once) and "align" (scan alignment, veins, upgrades of older cases). Admission and deletion
check each other under one lock, so a case is never deleted while it has queued or running work, and no new
work is admitted for a case while it is being deleted. A marker file in the case folder records a running stage;
one left behind by a closed app is reported as "interrupted" on the next start.
"""
import contextlib
import itertools
import os
import queue
import threading
import traceback

ACTIVE = ("queued", "running")
INTERRUPTED = "Stopped when BodyGlass was closed. Add the scan again to finish it."


class Busy(Exception):
    """The case has this stage queued or running, is being deleted, or no longer exists."""


class Jobs:
    def __init__(self, cases_root, lanes=None):
        self.root, self.lock, self.ids = cases_root, threading.Lock(), itertools.count(1)
        self.jobs, self.deleting, self.sync = {}, set(), {}   # (caseId, stage) -> latest job; cases being deleted; open writers
        self.lanes = {}
        for lane, workers in (lanes or {"heavy": 1, "align": 1}).items():
            self.lanes[lane] = queue.Queue()
            for _ in range(workers):
                threading.Thread(target=self._worker, args=(self.lanes[lane],), daemon=True).start()
        for cid in sorted(os.listdir(cases_root)) if os.path.isdir(cases_root) else []:
            for f in os.listdir(os.path.join(cases_root, cid)) if os.path.isdir(os.path.join(cases_root, cid)) else []:
                if f.startswith(".job-"):   # nothing runs yet, so this stage was cut off by a crash or quit
                    self.jobs[(cid, f[5:])] = self._record(cid, f[5:], status="interrupted", msg=INTERRUPTED)

    def _record(self, cid, kind, **kw):
        return dict({"id": next(self.ids), "caseId": cid, "kind": kind, "status": "queued", "pct": 0,
                     "msg": "Waiting for other work to finish", "error": None}, **kw)

    def _marker(self, job):
        return os.path.join(self.root, job["caseId"], ".job-" + job["kind"])

    def submit(self, cid, kind, fn, lane, new_case=False):
        """Queue fn(up) for a case. up(pct, msg, **fields) updates this job only. Raises Busy."""
        with self.lock:
            old = self.jobs.get((cid, kind))
            if (cid in self.deleting or (old and old["status"] in ACTIVE)
                    or not (new_case or os.path.exists(os.path.join(self.root, cid, "manifest.json")))):
                raise Busy(cid)
            job = self.jobs[(cid, kind)] = self._record(cid, kind)
        self.lanes[lane].put((job, fn))
        return job

    def quietly(self, *a, **kw):
        """submit() for follow-up work: already queued (or the case is gone) is fine."""
        with contextlib.suppress(Busy):
            self.submit(*a, **kw)

    def _set(self, job, **kw):
        with self.lock:
            job.update(kw)

    def _worker(self, q):
        while True:
            job, fn = q.get()
            self._set(job, status="running", msg="Starting")

            def up(pct=None, msg=None, **kw):   # bound to this job; a late call after it ended changes nothing
                with self.lock:
                    if job["status"] == "running":
                        job.update(**({"pct": pct} if pct is not None else {}), **({"msg": msg} if msg else {}), **kw)
            try:
                os.makedirs(os.path.dirname(self._marker(job)), exist_ok=True)
                open(self._marker(job), "w").close()
                fn(up)
                with self.lock:
                    if job["status"] == "running":   # fn may have reported its own end (error, unavailable, none)
                        job.update(status="done", pct=100)
            except Exception as e:
                traceback.print_exc()
                self._set(job, status="error", error=str(e) or type(e).__name__)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(self._marker(job))

    def snapshot(self):
        """{caseId: {stage: job}} for the viewer; every case's stages, never another case's."""
        out = {}
        with self.lock:
            for (cid, kind), job in self.jobs.items():
                out.setdefault(cid, {})[kind] = dict(job)
        return out

    def busy(self, cid):
        return (cid in self.deleting or self.sync.get(cid, 0) > 0
                or any(j["caseId"] == cid and j["status"] in ACTIVE for j in self.jobs.values()))

    @contextlib.contextmanager
    def writing(self, cid):
        """A short write in the caller's thread (rename, skin update): refused while the case is being deleted."""
        with self.lock:
            if cid in self.deleting:
                raise Busy(cid)
            self.sync[cid] = self.sync.get(cid, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self.sync[cid] -= 1

    def delete(self, cid, fn):
        """Run fn() (the removal) only if the case has no queued or running work; raises Busy otherwise."""
        with self.lock:
            if self.busy(cid):
                raise Busy(cid)
            self.deleting.add(cid)
        done = False
        try:
            fn()
            done = True
        finally:
            with self.lock:
                self.deleting.discard(cid)
                for key in [k for k in self.jobs if k[0] == cid and done]:
                    del self.jobs[key]

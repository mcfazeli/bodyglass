"""Run structure labelling (FastSurfer) on the public benchmark cases and record timings / volumes.

usage: python tools/run_phase2_bench.py SRC_CASES DST_CASES caseId[,caseId...]"""
import json, os, resource, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bodyglass import phase2

src, dst, ids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
names = {b["caseId"]: b["dataset"] for b in json.load(open(os.path.join(src, "benchmark.json")))}
out_path = os.path.join(dst, "phase2_benchmark.json")
results = json.load(open(out_path)) if os.path.exists(out_path) else []
for cid in ids:
    d = os.path.join(dst, cid)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(src, cid), d)
    t = time.time()
    r = phase2.build(d, lambda p, m: print(f"  [{cid}] {p:3.0f}% {m}", flush=True), device="cpu")
    mf = json.load(open(os.path.join(d, "manifest.json")))
    vol = {}
    for s in mf["structures"]:
        vol[s["key"]] = round(vol.get(s["key"], 0) + s["volume_ml"], 1)
    res = {"dataset": names.get(cid, cid), "caseId": cid, "timing": r["timing"], "wall": round(time.time() - t, 1),
           "child_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
           "seg_total_ml": r["totalSegVolumeMl"], "n_structures": len(mf["structures"]),
           "faces_total": sum(s["faces"] for s in mf["structures"]),
           "bytes_total": sum(os.path.getsize(os.path.join(d, s["file"])) for s in mf["structures"]),
           "volumes_ml": vol}
    results = [x for x in results if x["caseId"] != cid] + [res]
    json.dump(results, open(out_path, "w"), indent=1)
    print("DONE", json.dumps(res), flush=True)

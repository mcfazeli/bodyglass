"""Build the bundled comparison brain at install time (no network at runtime).

Input: the public OpenNeuro ds003097 (AOMIC-ID1000, CC0) sub-0001 T1w NIfTI, downloaded by
install.sh with a SHA-256 check. It is converted to DICOM in a temporary folder, run through the
same pipeline as your scans (Phase 1 + structures), and stored in example/ next to the app.
Skin and skull meshes are deleted (only the brain is needed for comparison).

usage: python tools/build_example.py NIFTI [--out example]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from bodyglass import phase2, pipeline  # noqa: E402

INFO = {"name": "Example brain (public, OpenNeuro ds003097 AOMIC-ID1000, sub-0001)",
        "license": "CC0 1.0 (public domain)", "url": "https://openneuro.org/datasets/ds003097",
        "credit": "Snoek et al., AOMIC (2021), Scientific Data. Healthy adult volunteer, defaced by the dataset authors.",
        "note": "One healthy adult volunteer. Brains differ a lot between healthy people; this is not a normal range."}

ap = argparse.ArgumentParser(); ap.add_argument("nifti"); ap.add_argument("--out", default=os.path.join(HERE, "example"))
a = ap.parse_args()
cases = os.path.join(a.out, "cases")
if os.path.exists(os.path.join(a.out, "example.json")):
    print("example already built"); sys.exit(0)
os.makedirs(cases, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="bv_example_")
try:
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "nifti_to_dicom.py"), a.nifti, tmp, "--desc", "T1 3D TFE",
                    "--manufacturer", "Philips", "--source", "OpenNeuro ds003097 sub-0001 (CC0)"], check=True)
    cid = pipeline.run(tmp, cases, lambda p, m: print(f"  {p:3.0f}% {m}", flush=True))
    phase2.build(os.path.join(cases, cid), lambda p, m: print(f"  structures {p:3.0f}% {m}", flush=True))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
cd = os.path.join(cases, cid)
mf = json.load(open(os.path.join(cd, "manifest.json")))
for L in [L for L in mf["layers"] if L["id"] in ("skin", "skull")]:
    os.remove(os.path.join(cd, L["file"]))
mf["layers"] = [L for L in mf["layers"] if L["id"] not in ("skin", "skull")]
mf["series"]["all"] = []
for f in ("volume.u8", "parcels.u8", "brainzone.u8", "series.json"):   # not needed for comparison
    if os.path.exists(os.path.join(cd, f)):
        os.remove(os.path.join(cd, f))
shutil.rmtree(os.path.join(cd, "fastsurfer"), ignore_errors=True)
mf.pop("parcels", None)
json.dump(mf, open(os.path.join(cd, "manifest.json"), "w"), indent=1)
json.dump(dict(INFO, caseId=cid), open(os.path.join(a.out, "example.json"), "w"), indent=1)
print("EXAMPLE_OK", cid)

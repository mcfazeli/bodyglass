"""A 3D vein tree traced from the aligned SWI scan (on this computer).

Veins carry oxygen-poor blood, which is dark on SWI. A Frangi 'vesselness' filter (Hessian
eigenvalues at several scales) highlights dark tubes and suppresses blobs (iron-rich nuclei)
and sheets (brain edge). Inside the brain zone and the SWI coverage the strongest responses are
kept with a hysteresis threshold, small specks are removed, and the result is meshed (veins.glb).
Cached: skipped if veins.glb exists for the same SWI file."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

SIGMAS = (1.0, 1.5, 2.0)   # mm; the SWI has 1.15 x 1.15 x 2 mm voxels, smaller scales mostly trace noise


def _load(case_dir, mf, name):
    nx, ny, nz = mf["volume"]["dims"]
    return np.fromfile(os.path.join(case_dir, name), np.uint8).reshape(nz, ny, nx)


def trace(swi, cover, zone, min_voxels=100):
    """Return (bool vessel mask, stats) on the same grid."""
    from skimage.filters import apply_hysteresis_threshold, frangi
    roi = ndi.binary_erosion(cover, iterations=3) & zone
    if roi.sum() < 1000:
        return None, {"reason": "SWI covers too little of the brain"}
    zz, yy, xx = np.nonzero(roi)
    sl = tuple(slice(max(0, a.min() - 4), a.max() + 5) for a in (zz, yy, xx))
    f = swi[sl].astype(np.float32)
    r = roi[sl]
    fill = np.median(f[r])
    f = np.where(r | (f > 0), f, fill)          # no-data -> neutral, so coverage edges are not 'vessels'
    v = frangi(f, sigmas=SIGMAS, black_ridges=True, alpha=0.5, beta=0.5, gamma=None)
    v[~r] = 0
    vals = v[r]
    hi, lo = np.percentile(vals, 99.7), np.percentile(vals, 98.5)   # tuned on QTAB: lower picks up sulcal CSF edges
    m = apply_hysteresis_threshold(v, lo, hi) & r
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_voxels
        keep[0] = False
        m = keep[lab]
    out = np.zeros(swi.shape, bool)
    out[sl] = m
    return out, {"components": int(ndi.label(m, structure=np.ones((3, 3, 3)))[1]), "voxels": int(m.sum()),
                 "roiMl": round(float(roi.sum()) / 1000, 1)}


def build(case_dir, progress=lambda pct, msg: None):
    from . import manifest_io
    from .meshes import export_glb, mask_to_mesh
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    seqs = {q["kind"]: q for q in mf.get("sequences", [])}
    src = seqs.get("swi") or seqs.get("mip")
    if not src or not mf.get("brainZone"):
        return None
    out = os.path.join(case_dir, "veins.glb")
    if mf.get("veins") and os.path.exists(out) and mf["veins"].get("source") == src["file"]:
        return mf["veins"]
    t0 = time.time()
    swi = _load(case_dir, mf, src["file"])
    zone = _load(case_dir, mf, mf["brainZone"]) > 0
    mask, stats = trace(swi, swi > 0, zone)
    if mask is None or mask.sum() < 200:
        info = {"error": stats.get("reason", "no veins found"), "source": src["file"]}
        manifest_io.update(case_dir, {"veins": info})
        return info
    sp = mf["volume"]["spacing"][0]
    # thicken by a voxel so 1-voxel veins survive smoothing; the surface sits near the vessel wall
    mesh = mask_to_mesh(ndi.binary_dilation(mask), mf["volume"]["origin"], sp, sigma=0.6,
                        target_faces=180_000, smooth_iters=6)
    size = export_glb(mesh, out, "veins")
    info = {"file": "veins.glb", "source": src["file"], "sourceName": src["name"], "faces": int(len(mesh.faces)),
            "bytes": size, "seconds": round(time.time() - t0, 1), "coveragePct": src.get("coveragePct"), **stats}
    manifest_io.update(case_dir, {"veins": info})
    return info


if __name__ == "__main__":   # python -m bodyglass.veins CASE_DIR [--force]
    import sys
    if "--force" in sys.argv:
        p = os.path.join(sys.argv[1], "veins.glb")
        if os.path.exists(p):
            os.remove(p)
    print(build(sys.argv[1]))

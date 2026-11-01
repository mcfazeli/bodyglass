"""Other MRI sequences (FLAIR, T2, SWI, DWI/ADC...) aligned to the T1, on this computer.

* pipeline.run() stores which series is which kind in <case>/series.json (metadata only).
* build(case_dir) loads each one, rigidly registers it to the T1 with mutual information
  (SimpleITK), resamples it onto the viewer's 1 mm grid and writes overlays/<kind>.u8
  (0 = no data there). Results are cached: a kind that already has a file is skipped.
* Series from the same acquisition (SWI / mIP / magnitude; DWI trace / ADC) share one transform.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .volume import load_series

SEQ_INFO = {   # name shown in the UI, colour map, which kind's transform it reuses
    "flair":   dict(name="FLAIR", cmap="hot"),
    "t2":      dict(name="T2", cmap="cool"),
    "tse":     dict(name="T2 TSE", cmap="cool"),
    "swi_mag": dict(name="SWI magnitude", cmap="grey"),
    "swi":     dict(name="SWI", cmap="ink", via="swi_mag"),
    "mip":     dict(name="SWI mIP", cmap="ink", via="swi_mag"),
    "dwi":     dict(name="DWI", cmap="hot"),
    "adc":     dict(name="ADC", cmap="adc", via="dwi"),
}
REG_ORDER = ["flair", "t2", "tse", "swi_mag", "swi", "mip", "dwi", "adc"]
ADC_MAX = 3000.0   # u8 255 == 3000e-6 mm^2/s


def _fixed_image(case_dir, mf):
    import SimpleITK as sitk
    nx, ny, nz = mf["volume"]["dims"]
    v = np.fromfile(os.path.join(case_dir, "volume.u8"), np.uint8).reshape(nz, ny, nx).astype(np.float32)
    img = sitk.GetImageFromArray(v)
    o = mf["volume"]["origin"]
    img.SetOrigin((-o[0], -o[1], o[2]))      # RAS grid expressed in LPS (DICOM patient space)
    img.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    img.SetSpacing([float(mf["volume"]["spacing"][0])] * 3)
    return img, v


def _moving_image(files):
    import SimpleITK as sitk
    vol, aff = load_series(files)
    img = sitk.GetImageFromArray(vol.astype(np.float32))
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    d = aff[:3, :3] / sp
    img.SetSpacing([float(x) for x in sp])
    img.SetDirection([float(x) for x in d.ravel()])
    img.SetOrigin([float(x) for x in aff[:3, 3]])
    return img


def _fixed_mask(v):
    """Head region of the T1 (ignores noisy air around MP2RAGE scans)."""
    from scipy import ndimage as ndi
    import SimpleITK as sitk
    from .segment import largest_cc
    sm = ndi.gaussian_filter(v, 2)
    m = ndi.binary_fill_holes(largest_cc(sm > max(25.0, np.percentile(sm[sm > 5], 40) * 0.5)))
    return sitk.GetImageFromArray(m.astype(np.uint8))


def register(fixed, moving, fmask=None, same_frame=True):
    """Rigid (6 dof) Mattes mutual-information registration. Returns (transform, info).

    shortcut: rigid only; scans with patient motion between series or strong EPI distortion (DWI) keep
    a residual of a few mm. Add affine/B-spline only if overlays visibly drift."""
    import SimpleITK as sitk
    thick = max(moving.GetSpacing()) > 3.0   # 2D scans with thick slices: more samples, longer search
    t0 = time.time()
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY if same_frame
        else sitk.CenteredTransformInitializerFilter.MOMENTS)
    if same_frame:   # scanner coordinates already agree: start from the header (identity), keep the centre
        init = sitk.Euler3DTransform(init.GetCenter(), 0, 0, 0, (0, 0, 0))
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25 if thick else 0.10, seed=7)
    if fmask is not None:
        R.SetMetricFixedMask(fmask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=500 if thick else 200,
                                               relaxationFactor=0.6, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([3, 1.5, 0.5] if thick else [2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    before = R.MetricEvaluate(fixed, moving)
    tx = R.Execute(fixed, moving)
    after = R.GetMetricValue()
    e = sitk.Euler3DTransform(tx.GetNthTransform(0) if hasattr(tx, "GetNthTransform") else tx)
    start = sitk.Euler3DTransform(init)
    c = start.GetCenter()   # how far the fit moved away from its starting guess (header or centre of mass)
    shift = float(np.linalg.norm(np.subtract(e.TransformPoint(c), start.TransformPoint(c))))
    rot = float(np.degrees(max(abs(a - b) for a, b in zip((e.GetAngleX(), e.GetAngleY(), e.GetAngleZ()),
                                                          (start.GetAngleX(), start.GetAngleY(), start.GetAngleZ())))))
    info = {"metricBefore": round(before, 4), "metric": round(after, 4), "shiftMm": round(shift, 2),
            "rotationDeg": round(rot, 2), "iterations": R.GetOptimizerIteration(),
            "seconds": round(time.time() - t0, 1), "method": "rigid MI"}
    # A worse metric (Mattes MI: lower is better) or a big jump from the start means the search slid off
    # (seen with thin slabs, where MI can 'improve' while drifting 80 mm): keep the starting guess.
    if after > before or shift > 20 or rot > 15:
        info.update(method="scanner coordinates (registration rejected)" if same_frame
                    else "centre of mass only (registration rejected)", rejected=True)
        e = start
    return e, info


def _to_u8(arr, cover, kind):
    if kind == "adc":
        u = np.clip(arr / ADC_MAX * 254, 0, 254) + 1
        rng = None
    else:
        vals = arr[cover & (arr > 0)]
        lo, hi = (np.percentile(vals, [1, 99.5]) if vals.size > 1000 else (0, 1))
        lo = 0.0 if kind in ("swi", "mip", "swi_mag") else lo
        u = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 254 + 1
        rng = [float(lo), float(hi)]
    u = np.where(cover, u, 0).astype(np.uint8)
    return u, rng


def build(case_dir, progress=lambda pct, msg: None):
    import SimpleITK as sitk
    sj = os.path.join(case_dir, "series.json")
    mfp = os.path.join(case_dir, "manifest.json")
    if not os.path.exists(sj):
        return []
    with open(sj) as fh:
        found = json.load(fh)
    with open(mfp) as fh:
        mf = json.load(fh)
    kinds = [k for k in REG_ORDER if k in found]
    _ensure_zone(case_dir, mf)
    os.makedirs(os.path.join(case_dir, "overlays"), exist_ok=True)
    cache = os.path.join(case_dir, "overlays", "sequences.json")   # source of truth for the cache
    if os.path.exists(cache):
        with open(cache) as fh:
            prev = json.load(fh)
    else:
        prev = mf.get("sequences", [])
    done = {s["kind"]: s for s in prev if "file" in s}
    if done and mf.get("sequences") != list(done.values()):
        _save(case_dir, done)
    todo = [k for k in kinds if k not in done or not os.path.exists(os.path.join(case_dir, done[k]["file"]))]
    if not todo:
        return list(done.values())
    fixed, fv = _fixed_image(case_dir, mf)
    fmask = _fixed_mask(fv)
    fmask.CopyInformation(fixed)
    transforms = {}
    for n, kind in enumerate(todo):
        info_k = SEQ_INFO[kind]
        rec = found[kind]
        progress(int(100 * n / len(todo)), f"Aligning {info_k['name']} to the T1")
        try:
            moving = _moving_image(rec["files"])
        except Exception as e:
            done[kind] = {"kind": kind, "error": f"could not read: {e}"}
            continue
        via = info_k.get("via")
        if via and via in transforms:
            tx, info = transforms[via][0], dict(transforms[via][1], method=f"same transform as {SEQ_INFO[via]['name']}", seconds=0)
        else:
            same = bool(rec.get("frameOfReference")) and rec.get("frameOfReference") == found.get("_t1", {}).get("frameOfReference")
            mv = moving
            if kind == "adc":
                mv = sitk.Clamp(moving, sitk.sitkFloat32, 0, 3500)
            tx, info = register(fixed, mv, fmask, same_frame=same)
        transforms[kind] = (tx, info)
        res = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        ones = sitk.Image(moving.GetSize(), sitk.sitkUInt8) + 1
        ones.CopyInformation(moving)
        cover = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, tx, sitk.sitkNearestNeighbor, 0)) > 0
        arr = sitk.GetArrayFromImage(res)
        u8, rng = _to_u8(arr, cover, kind)
        fn = f"overlays/{kind}.u8"
        u8.tofile(os.path.join(case_dir, fn))
        sp = moving.GetSpacing()
        done[kind] = {"kind": kind, "name": info_k["name"], "file": fn, "cmap": info_k["cmap"],
                      "description": rec.get("description", ""), "reason": rec.get("reason", ""),
                      "voxelMm": [round(float(x), 2) for x in sp], "coveragePct": round(100 * float(cover.mean()), 1),
                      "align": info, "range": ([0, ADC_MAX] if kind == "adc" else rng)}
        _save(case_dir, done)
    progress(100, "Scans aligned")
    return [done[k] for k in REG_ORDER if k in done]


def _ensure_zone(case_dir, mf):
    """Older cases: derive brainzone.u8 from the structure labels (brain + 4 mm) if available."""
    from scipy import ndimage as ndi
    from . import manifest_io
    zp = os.path.join(case_dir, "brainzone.u8")
    if os.path.exists(zp):
        return
    lv = mf.get("labelVolume")
    if not lv or not os.path.exists(os.path.join(case_dir, lv["file"])):
        return
    nx, ny, nz = mf["volume"]["dims"]
    lab = np.fromfile(os.path.join(case_dir, lv["file"]), np.uint8).reshape(nz, ny, nx) > 0
    ndi.binary_dilation(ndi.binary_closing(lab, iterations=2), iterations=4).astype(np.uint8).tofile(zp)
    manifest_io.update(case_dir, {"brainZone": "brainzone.u8"})


def _save(case_dir, done):
    from . import manifest_io
    seqs = [done[k] for k in REG_ORDER if k in done and "file" in done[k]]
    with open(os.path.join(case_dir, "overlays", "sequences.json"), "w") as fh:
        json.dump(seqs, fh, indent=1)
    manifest_io.update(case_dir, {"sequences": seqs})


def write_series_json(case_dir, series, t1):
    """Called by pipeline.run(): remember which series is which kind (headers only)."""
    from .dicom_scan import pick_sequences
    picks = pick_sequences(series, t1)
    out = {"_t1": {"uid": t1.uid, "files": t1.files, "frameOfReference": t1.meta.get("FrameOfReferenceUID")}}   # files: neck.py re-reads the neck part
    for k, p in picks.items():
        s = p["series"]
        out[k] = {"uid": s.uid, "files": s.files, "description": s.meta.get("SeriesDescription", ""),
                  "reason": p["reason"], "alternatives": p["alternatives"],
                  "frameOfReference": s.meta.get("FrameOfReferenceUID")}
    with open(os.path.join(case_dir, "series.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return picks

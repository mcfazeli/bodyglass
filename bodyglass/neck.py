"""Neck exam (v1): a second DICOM folder (cervical spine / neck MRI) shown together with the brain model.

1. pick the best sagittal 3D T1/T2 neck series by metadata only (like dicom_scan.score_series),
2. TotalSegmentator (Apache-2.0, run locally, usage statistics off, network blocked): vertebrae_mr on both
   exams, total_mr (spinal cord, discs) on the neck,
3. rigid fit neck -> brain T1 from the C1-C3 vertebrae seen in both exams (landmarks, not image intensity),
   refused when the shapes still disagree by more than MAX_RMS_MM,
4. the brain grid is extended downwards: neck/volume.u8 (brain T1, neck exam below it) and neck/labels.u8
   (brain labels + neck labels 200+, brain labels win), one mesh per neck structure.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from . import manifest_io
from .dicom_scan import _text, scan_folder
from .meshes import export_glb, mask_to_mesh
from .phase2 import STRUCT_DEFS, resample_labels
from .volume import load_series, to_ras_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_HOME = os.environ.get("BODYGLASS_TOTALSEG", os.path.join(ROOT, "models", "totalseg"))
FIT_LEVELS = ["C1", "C2", "C3"]               # posture changes less here than lower in the neck
SHOW_LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"]
MAX_RMS_MM = 3.0
MAX_PAD_MM = 250                              # how far the slices may extend below the brain scan
CODE = {f"vertebra_{l.lower()}": 200 + i for i, l in enumerate(SHOW_LEVELS)}
CODE.update(intervertebral_discs=208, spinal_cord=209)
LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def available() -> bool:
    import importlib.util
    return os.path.exists(os.path.join(TS_HOME, "ready")) and importlib.util.find_spec("totalsegmentator") is not None


# ---------------------------------------------------------------- 1. series pick (metadata only)
def score_neck(s):
    """(score, reasons, excluded) for a neck/spine exam series. Names are unreliable, so orientation,
    3D acquisition and voxel size count most."""
    m, text = s.meta, _text(s.meta)
    tokens = set(re.split(r"[^A-Z0-9]+", text))
    itype = [str(x).upper() for x in (m.get("ImageType") or [])]
    ss = m.get("ScanningSequence")
    scan_seq = [str(x).upper() for x in ss] if isinstance(ss, list) else str(ss or "").upper().split("\\")
    score, why, excluded = 0.0, [], None

    def add(pts, reason):
        nonlocal score
        score += pts
        why.append(f"{'+' if pts >= 0 else ''}{pts:g} {reason}")

    if (m.get("Modality") or "MR") != "MR":
        excluded = "not MR"
    elif m.get("noGeometry"):
        excluded = "multi-frame file without slice positions"
    elif tokens & {"LOCALIZER", "SCOUT", "LOC", "SURVEY", "AAHEAD"}:
        excluded = "localizer"
    elif "EP" in scan_seq or tokens & {"DWI", "ADC", "TRACEW", "DTI"} or "ADC" in itype or "DIFFUSION" in itype:
        excluded = "diffusion / echo-planar"
    elif "P" in itype or "PHASE" in itype:
        excluded = "phase image"
    elif len(s.files) < 20:
        excluded = "too few images"
    iop = m.get("ImageOrientationPatient")
    if iop and len(iop) == 6:
        normal = np.abs(np.cross(np.array(iop[:3], float), np.array(iop[3:], float)))
        plane = ["sagittal", "coronal", "axial"][int(np.argmax(normal))]
        add({"sagittal": 25, "coronal": 0, "axial": -15}[plane], f"{plane} images")
    if m.get("MRAcquisitionType") == "3D":
        add(30, "MRAcquisitionType=3D")
    elif m.get("MRAcquisitionType") == "2D":
        add(-10, "2D acquisition")
    if tokens & {"T2", "T2W", "T1", "T1W", "SPC", "SPACE", "CUBE", "VISTA", "TSE", "FSE", "MPRAGE"} \
            or any(k in str(m.get("SequenceName") or "").upper() for k in ("SPC", "TSE", "TFL", "FSE")):
        add(15, "T1/T2 structural sequence")
    if tokens & {"STIR", "TIRM", "FS", "FATSAT", "SPAIR", "DIXON", "MEDIC", "GRE", "ME", "T2STAR", "SWI"}:
        add(-15, "fat-suppressed / gradient-echo variant (less clear bone edges)")
    if "DERIVED" in itype or "SECONDARY" in itype or "MPR" in tokens or "REFORMAT" in text:
        add(-40, "derived/reformatted image")
    elif "ORIGINAL" in itype:
        add(5, "ORIGINAL image")
    ps = m.get("PixelSpacing") or [1.0, 1.0]
    st = float(m.get("SliceThickness") or 0) or None
    if st:
        vs = [float(ps[0]), float(ps[1]), st]
        if max(vs) <= 1.3:
            add(20, f"fine voxels {vs[0]:.2g}x{vs[1]:.2g}x{vs[2]:.2g} mm")
        elif max(vs) > 3:
            add(-10, f"thick slices ({max(vs):.2g} mm)")
    add(min(len(s.files), 200) / 10.0, f"{len(s.files)} images")
    return score, why + ([f"excluded: {excluded}"] if excluded else []), excluded


SPINE_WORDS = ("SPINE", "CSPINE", "C-SPINE", "CERV", "NECK", "SPINAL", "HWS")


def is_spine(s):
    """Series labelled as a spine/neck scan. Any HEAD/BRAIN word vetoes it, so a 'head and C-spine' study or a
    'T1w head-neck' series stays a head scan."""
    m = s.meta
    text = " ".join(str(m.get(k) or "") for k in ("BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName")).upper()
    return any(w in text for w in SPINE_WORDS) and not any(w in text for w in ("HEAD", "BRAIN"))   # "T1w head-neck" is a head scan


def route(series):
    """'Add scan' sorting: {'brain': T1 series or None, 'neck': neck series or None}. A brain exam's own sagittal
    T2 is never taken as a neck: next to a brain T1, only spine-labelled series are neck candidates (Study/Series
    description or BodyPartExamined); with no brain T1, every series is.
    shortcut: an unlabelled neck exam exported together with a brain exam is not found; add its folder on its own.
    shortcut: an unlabelled neck-only 3D series with a T1-like name can still be taken as a brain T1."""
    from .dicom_scan import pick_t1
    spine = [s for s in series if is_spine(s)]
    brain = pick_t1([s for s in series if s not in spine])
    pool = spine if brain else series
    return {"brain": brain, "neck": pick_neck(pool) if pool else None}


def pick_neck(series):
    """Best neck series or None; also stores score/reasons on each series for the 'Why this series?' list."""
    best = None
    for s in series:
        s.score, s.reasons, s.excluded = score_neck(s)
        if not s.excluded and (best is None or s.score > best.score):
            best = s
    return best


# ---------------------------------------------------------------- 2. TotalSegmentator (local subprocess)
def _telemetry_off():
    """TotalSegmentator creates a config with usage statistics ON if none exists: make sure ours says off."""
    p = os.path.join(TS_HOME, "config.json")
    cfg = {}
    if os.path.exists(p):
        with open(p) as fh:
            cfg = json.load(fh)
    if cfg.get("send_usage_stats", True):
        cfg.update(totalseg_id=cfg.get("totalseg_id", "totalseg_local"), send_usage_stats=False,
                   prediction_counter=cfg.get("prediction_counter", 0))
        with open(p, "w") as fh:
            json.dump(cfg, fh)


def run_totalseg(inp, out, task, device, roi=None, log=None):
    """One TotalSegmentator run; outputs are kept so an interrupted neck build resumes."""
    if os.path.exists(out):
        return out
    _telemetry_off()
    # a dead proxy: if anything in TotalSegmentator tried the network (stats, weights) it fails instead
    env = dict(os.environ, TOTALSEG_HOME_DIR=TS_HOME, PYTORCH_ENABLE_MPS_FALLBACK="1", NO_PROXY="", no_proxy="",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9")
    exe = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")
    # run the script through this python: its #! line holds the install path, which breaks if the folder is moved/renamed
    cmd = [sys.executable, exe, "-i", inp, "-o", out + ".part.nii.gz", "--ml", "--task", task, "--device", device]
    if roi:
        cmd += ["--roi_subset", *roi]
    with open(log or os.devnull, "a") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 and device != "cpu":   # MPS is new for nnU-Net: retry on the CPU
        return run_totalseg(inp, out, task, "cpu", roi, log)
    if rc != 0:
        raise RuntimeError(f"Neck labelling failed (TotalSegmentator {task}, exit {rc}); see {log}")
    os.replace(out + ".part.nii.gz", out)
    return out


def _labels(path, task):
    """{name: (N,3) RAS mm points} from a TotalSegmentator multi-label file."""
    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map
    img = nib.load(path)
    lab = np.asarray(img.dataobj).astype(np.int16)
    out = {}
    for code, name in class_map[task].items():
        ijk = np.argwhere(lab == code)
        if len(ijk) >= 50:
            out[name] = ijk @ img.affine[:3, :3].T + img.affine[:3, 3]
    return out, lab, img.affine


def _brain_t1(case_dir, bvol, origin, sp):
    """The brain T1 on the viewer grid before air cleaning (which blanks the neck more than ~2 cm below the
    brain): re-read from its DICOM files if they are still there, else the viewer volume (finds fewer vertebrae)."""
    try:
        with open(os.path.join(case_dir, "series.json")) as fh:
            files = json.load(fh)["_t1"]["files"]
        grid, o, _ = to_ras_grid(*load_series(files), spacing=sp)
        if grid.shape == bvol.shape and np.allclose(o, origin, atol=1e-3):
            return grid, "dicom"
    except (OSError, KeyError, ValueError, IndexError):
        pass
    return bvol, "viewer"


# ---------------------------------------------------------------- 3. landmark fit
def kabsch(P, Q):
    """Rigid R, t minimising |R P + t - Q| (rows are points)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def fit_rigid(mov, fix, mov_inside, fix_inside, iters=60):
    """Rigid transform (4x4) taking neck-exam points `mov` onto brain-exam points `fix`.

    mov / fix: {level: (N,3) points in mm}, the same vertebrae in both exams. Start: the vertebra centroids
    (3 nearly collinear points leave the turn about the spine open), then closest-point refinement per
    vertebra, using only the parts both exams can see (mov_inside / fix_inside: points -> bool mask in
    the neck / brain frame), so a thin slab is not pulled into the full vertebra. Returns (T, rms_mm)."""
    lv = [l for l in mov if l in fix]
    R, t = kabsch(np.array([mov[l].mean(0) for l in lv]), np.array([fix[l].mean(0) for l in lv]))
    for _ in range(iters):
        P, Q = [], []
        for l in lv:
            m = mov[l][fix_inside(mov[l] @ R.T + t)]
            f = fix[l][mov_inside((fix[l] - t) @ R)]
            if len(m) < 20 or len(f) < 20:
                continue
            mw = m @ R.T + t
            P.append(m); Q.append(f[cKDTree(f).query(mw)[1]])        # neck -> nearest brain point
            P.append(m[cKDTree(mw).query(f)[1]]); Q.append(f)        # brain -> nearest neck point
        if not P:
            return None, float("inf")
        P, Q = np.concatenate(P), np.concatenate(Q)
        R2, t2 = kabsch(P, Q)
        done = np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-4)
        R, t = R2, t2
        if done:
            break
    rms = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1))))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T, rms


def _inside(affine, shape, margin=2):
    inv = np.linalg.inv(affine)
    hi = np.array(shape) - 1 - margin
    return lambda P: np.all(((P @ inv[:3, :3].T + inv[:3, 3]) >= margin) & ((P @ inv[:3, :3].T + inv[:3, 3]) <= hi), 1)


# ---------------------------------------------------------------- 4. labels on the extended grid
def merge_labels(brain, neck):
    """Brain (FastSurfer) labels win where both exams have a label; neck labels fill the rest."""
    return np.where(brain > 0, brain, neck).astype(np.uint8)


def build(case_dir, folder, progress=lambda p, m: None, device=None, uid=None):
    t0 = time.time()
    work = os.path.join(case_dir, "neck"); os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "totalseg.log")
    device = device or os.environ.get("BODYGLASS_DEVICE", "auto")
    if device == "auto":
        device = "mps" if platform.system() == "Darwin" else "cpu"
    import nibabel as nib
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    V = mf["volume"]; nx, ny, nz = V["dims"]; origin = np.array(V["origin"], float); sp = float(V["spacing"][0])

    progress(3, "Reading the neck exam's DICOM headers")
    series = scan_folder(folder)
    best = pick_neck(series)
    if uid:   # chosen by route()
        best = next((s for s in series if s.uid == uid), best)
    if best is None:
        raise RuntimeError("No usable neck series found in this folder (looking for a sagittal T1 or T2).")
    progress(8, f"Loading {best.meta.get('SeriesDescription') or 'neck series'} ({len(best.files)} images)")
    vol, aff = load_series(best.files)
    A = LPS2RAS @ aff                                   # neck voxel (i,j,k) -> RAS mm
    neck_nii = os.path.join(work, "neck.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0)), A), neck_nii)
    brain_nii = os.path.join(work, "brain.nii.gz")
    bvol = np.fromfile(os.path.join(case_dir, V["file"]), np.uint8).reshape(nz, ny, nx)
    B = np.diag([sp, sp, sp, 1.0]); B[:3, 3] = origin
    braw, bsrc = _brain_t1(case_dir, bvol, origin, sp)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(braw.transpose(2, 1, 0)), B), brain_nii)

    tag = hashlib.sha1(best.uid.encode()).hexdigest()[:8]   # outputs are kept per neck series (resume / replace)
    out = {"neck_vert": f"{tag}_vert", "neck_cord": f"{tag}_cord", "brain_vert": f"brain_vert_{bsrc}"}
    timing = {}
    for pct, (name, inp, task, roi) in zip((15, 40, 65), (
            ("neck_vert", neck_nii, "vertebrae_mr", None),
            ("neck_cord", neck_nii, "total_mr", ["spinal_cord", "intervertebral_discs"]),
            ("brain_vert", brain_nii, "vertebrae_mr", None))):
        progress(pct, {"neck_vert": "Finding the vertebrae in the neck exam",
                       "neck_cord": "Finding the spinal cord and discs",
                       "brain_vert": "Finding the upper vertebrae in the brain scan"}[name])
        t = time.time()
        run_totalseg(inp, os.path.join(work, out[name] + ".nii.gz"), task, device, roi, log)
        timing[name] = round(time.time() - t, 1)

    progress(85, "Lining up the neck with the brain scan (C1–C3)")
    nv, nlab, nA = _labels(os.path.join(work, out["neck_vert"] + ".nii.gz"), "vertebrae_mr")
    bv, _, _ = _labels(os.path.join(work, out["brain_vert"] + ".nii.gz"), "vertebrae_mr")
    mov = {l: nv[f"vertebrae_{l}"][::2] for l in FIT_LEVELS if f"vertebrae_{l}" in nv}
    fix = {l: bv[f"vertebrae_{l}"][::2] for l in FIT_LEVELS if f"vertebrae_{l}" in bv}
    shared = [l for l in FIT_LEVELS if l in mov and l in fix]
    T, rms = (fit_rigid({l: mov[l] for l in shared}, fix, _inside(A, vol.shape[::-1]), _inside(B, (nx, ny, nz)))
              if len(shared) >= 2 else (None, float("inf")))
    ok = T is not None and rms <= MAX_RMS_MM
    info = {"series": best.summary(), "levels": shared,
            "rmsMm": round(rms, 2) if np.isfinite(rms) else None, "limitMm": MAX_RMS_MM,
            "status": "ok" if ok else "refused", "device": device, "tool": "TotalSegmentator 2.18 (total_mr, vertebrae_mr)"}
    if not ok:
        info["reason"] = (f"Only {len(shared)} of the vertebrae C1–C3 were found in both scans (at least 2 are needed)."
                          if len(shared) < 2 else
                          f"The C1–C3 vertebrae still differ by {rms:.1f} mm after lining up (limit {MAX_RMS_MM:g} mm).")
        info["timing"] = dict(timing, total=round(time.time() - t0, 1))
        manifest_io.update(case_dir, {"neck": info})
        progress(100, "Neck exam could not be lined up")
        return info

    progress(88, "Placing the neck labels and slices")
    M = T @ A                                           # neck voxel -> brain RAS
    corners = np.array([[i, j, k, 1] for i in (0, vol.shape[2] - 1) for j in (0, vol.shape[1] - 1) for k in (0, vol.shape[0] - 1)]).T
    low = float((M @ corners)[2].min())
    pad = int(np.clip(np.ceil((origin[2] - low) / sp), 0, MAX_PAD_MM / sp))
    # shortcut: the grid only grows downwards; neck parts outside the brain scan's left-right / front-back box are cut off
    eorigin = origin - np.array([0, 0, pad * sp]); edims = (nx, ny, nz + pad)
    # neck labels in the neck exam's voxels: vertebrae C1-T1, discs, cord (codes 200+)
    lab = np.zeros(nlab.shape, np.int16)
    from totalsegmentator.map_to_binary import class_map
    inv_v = {n: c for c, n in class_map["vertebrae_mr"].items()}
    for l in SHOW_LEVELS:
        if f"vertebrae_{l}" in inv_v:
            lab[nlab == inv_v[f"vertebrae_{l}"]] = CODE[f"vertebra_{l.lower()}"]
    cimg = nib.load(os.path.join(work, out["neck_cord"] + ".nii.gz")); clab = np.asarray(cimg.dataobj).astype(np.int16)
    inv_t = {n: c for c, n in class_map["total_mr"].items()}
    for key in ("intervertebral_discs", "spinal_cord"):
        lab[(clab == inv_t[key]) & (lab == 0)] = CODE[key]
    neck_lab = resample_labels(nib.Nifti1Image(lab, T @ nA), edims, eorigin, sp)
    brain_lab = np.zeros((nz + pad, ny, nx), np.uint8)
    lv = mf.get("labelVolume")
    if lv and os.path.exists(os.path.join(case_dir, lv["file"])):
        brain_lab[pad:] = np.fromfile(os.path.join(case_dir, lv["file"]), np.uint8).reshape(nz, ny, nx)
    merged = merge_labels(brain_lab, neck_lab)
    merged.tofile(os.path.join(work, "labels.u8"))
    # slice picture: the brain T1 where the viewer shows it, the neck exam everywhere else (air-cleaned neck, below)
    # shortcut: the neck keeps its own contrast (e.g. T2: fluid bright), so there is a visible seam; fine for v1
    lo, hi = np.percentile(vol[vol > 0], [1, 99.5]) if (vol > 0).any() else (0, 1)
    v8 = (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.int16)
    ext = resample_labels(nib.Nifti1Image(np.ascontiguousarray(v8.transpose(2, 1, 0)), M), edims, eorigin, sp).astype(np.uint8)
    ext[pad:] = np.where(bvol > 0, bvol, ext[pad:])
    ext.tofile(os.path.join(work, "volume.u8"))

    progress(92, "Building 3D shapes of the vertebrae, discs and cord")
    defs = STRUCT_DEFS["structures"]
    structures = []
    for key, code in CODE.items():
        m = (merged == code)
        if m.sum() < 30:
            continue
        mesh = mask_to_mesh(m, eorigin, sp, sigma=0.7, target_faces=20_000, smooth_iters=10)
        if mesh is None:
            continue
        fn = f"neck/{key}.glb"
        export_glb(mesh, os.path.join(case_dir, fn), key)
        structures.append({"id": key, "key": key, "hemi": None, "group": "spine", "name": defs[key]["name"],
                           "color": defs[key]["color"], "file": fn, "faces": int(len(mesh.faces)),
                           "volume_ml": round(float(m.sum()) * sp ** 3 / 1000, 1), "centroid": [float(c) for c in mesh.centroid]})
    info.update(pad=pad, volumeFile="neck/volume.u8", labelFile="neck/labels.u8",
                labelIndex={str(c): k for k, c in CODE.items()}, structures=structures,
                group=STRUCT_DEFS["groups"]["spine"], structureInfo={k: defs[k]["description"] for k in CODE},
                transform=[[round(float(x), 5) for x in row] for row in T],
                timing=dict(timing, total=round(time.time() - t0, 1)))
    manifest_io.update(case_dir, {"neck": info})
    progress(100, "Neck ready")
    return info

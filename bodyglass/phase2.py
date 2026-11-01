"""Phase 2: labelled structures with FastSurfer (FastSurferVINN, --seg_only equivalent).

Runs FastSurferCNN/run_prediction.py (Apache-2.0, Deep-MI) locally on the case's 1 mm T1, then
turns the aparc.DKTatlas+aseg label map into one mesh per structure:
  * cortex lobes: the Phase 1 brain surface is split by the nearest cortical label, so the lobe
    meshes tile seamlessly and keep the nice surface,
  * deep structures, ventricles, white matter, cerebellum, brainstem: closed meshes from labels,
  * parcels.u8: nearest DKT gyrus per voxel, so a click on a lobe can name the gyrus.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy import ndimage as ndi

from . import manifest_io
from .meshes import export_glb, mask_to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
FASTSURFER = os.environ.get("BODYGLASS_FASTSURFER",
                            os.path.join(os.path.dirname(HERE), "third_party", "FastSurfer"))
with open(os.path.join(HERE, "structures.json")) as _fh:
    STRUCT_DEFS = json.load(_fh)

DKT = {2: "caudalanteriorcingulate", 3: "caudalmiddlefrontal", 5: "cuneus", 6: "entorhinal", 7: "fusiform",
       8: "inferiorparietal", 9: "inferiortemporal", 10: "isthmuscingulate", 11: "lateraloccipital",
       12: "lateralorbitofrontal", 13: "lingual", 14: "medialorbitofrontal", 15: "middletemporal",
       16: "parahippocampal", 17: "paracentral", 18: "parsopercularis", 19: "parsorbitalis",
       20: "parstriangularis", 21: "pericalcarine", 22: "postcentral", 23: "posteriorcingulate",
       24: "precentral", 25: "precuneus", 26: "rostralanteriorcingulate", 27: "rostralmiddlefrontal",
       28: "superiorfrontal", 29: "superiorparietal", 30: "superiortemporal", 31: "supramarginal",
       34: "transversetemporal", 35: "insula"}
LOBE = {}
for k, names in {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
                "precentral", "paracentral"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "transversetemporal",
                 "fusiform", "entorhinal", "parahippocampal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "insula": ["insula"],
}.items():
    for n in names:
        LOBE[n] = k
# closed-mesh structures: key -> (left labels, right labels) ; None = midline (unpaired)
ASEG = {
    "cerebral_wm": ([2], [41]), "thalamus": ([10], [49]), "caudate": ([11], [50]), "putamen": ([12], [51]),
    "pallidum": ([13], [52]), "accumbens": ([26], [58]), "hippocampus": ([17], [53]),
    "amygdala": ([18], [54]), "ventral_dc": ([28], [60]), "lateral_ventricle": ([4, 5, 31], [43, 44, 63]),
    "cerebellum": ([7, 8], [46, 47]), "third_ventricle": ([14], None), "fourth_ventricle": ([15], None),
    "brainstem": ([16], None),
}
MESH_OPTS = {
    "cerebral_wm": dict(sigma=0.8, target_faces=220_000, smooth_iters=10),
    "cerebellum": dict(sigma=0.8, target_faces=120_000, smooth_iters=10),
    "brainstem": dict(sigma=1.0, target_faces=40_000, smooth_iters=10),
}
DEFAULT_OPTS = dict(sigma=1.0, target_faces=25_000, smooth_iters=15)
HEMI_NAME = {"L": "Left", "R": "Right", None: ""}


def available() -> bool:
    ck = os.path.join(FASTSURFER, "checkpoints")
    return os.path.exists(os.path.join(FASTSURFER, "FastSurferCNN", "run_prediction.py")) and \
        all(os.path.exists(os.path.join(ck, f"aparc_vinn_{v}_v2.0.0.pkl")) for v in ("axial", "coronal", "sagittal"))


def run_fastsurfer(t1_path, workdir, progress=lambda p, m: None, device="auto"):
    """Returns (label nibabel image, seconds, log path)."""
    import nibabel as nib
    t1_path, workdir = os.path.abspath(t1_path), os.path.abspath(workdir)
    log = os.path.join(workdir, "fastsurfer.log")
    env = dict(os.environ, PYTHONPATH=FASTSURFER, PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [sys.executable, os.path.join(FASTSURFER, "FastSurferCNN", "run_prediction.py"),
           "--t1", t1_path, "--sid", "fs", "--sd", workdir, "--device", device,
           "--viewagg_device", "cpu", "--batch_size", "1", "--threads", str(os.cpu_count() or 4)]
    t0 = time.time()
    views = {"coronal": 0, "sagittal": 1, "axial": 2}
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=FASTSURFER, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            fh.write(line)
            m = re.search(r"Run (\w+) prediction", line)
            if m and m.group(1) in views:
                progress(10 + 25 * views[m.group(1)], f"Labelling structures: {m.group(1)} view")
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"FastSurfer failed (exit {rc}); see {log}")
    seg = nib.load(os.path.join(workdir, "fs", "mri", "aparc.DKTatlas+aseg.deep.mgz"))
    return seg, round(time.time() - t0, 1), log


def resample_labels(seg, dims, origin, sp):
    """Nearest-neighbour labels on the viewer's RAS grid [z,y,x]."""
    lab = np.asarray(seg.dataobj).astype(np.int16)
    inv = np.linalg.inv(seg.affine)
    nx, ny, nz = dims
    out = np.zeros((nz, ny, nx), np.int16)
    X, Y = np.meshgrid(origin[0] + np.arange(nx) * sp, origin[1] + np.arange(ny) * sp, indexing="xy")
    for zi in range(nz):
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, origin[2] + zi * sp), np.ones(X.size)])
        ijk = np.rint((inv @ pts)[:3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(lab.shape)[:, None]), axis=0)
        sl = np.zeros(X.size, np.int16)
        sl[ok] = lab[ijk[0, ok], ijk[1, ok], ijk[2, ok]]
        out[zi] = sl.reshape(X.shape)
    return out


def write_label_volume(case_dir, lab, structures):
    """labels.u8 on the viewer grid: value k = structures[k-1] (0 = none), for the 2D/3D slice overlay."""
    sid_index = {st["id"]: i + 1 for i, st in enumerate(structures)}
    code2sid = {}
    for code, name in DKT.items():
        code2sid[1000 + code] = f"{LOBE[name]}_L"; code2sid[2000 + code] = f"{LOBE[name]}_R"
    for key, (left, right) in ASEG.items():
        for hemi, labels in (("L", left), ("R", right)) if right else ((None, left),):
            for l in labels:
                code2sid[l] = f"{key}_{hemi}" if hemi else key
    lut = np.zeros(int(lab.max()) + 1 if lab.size else 1, np.uint8)
    for code, sid in code2sid.items():
        if code < len(lut) and sid in sid_index:
            lut[code] = sid_index[sid]
    out = lut[np.clip(lab, 0, len(lut) - 1)]
    out.tofile(os.path.join(case_dir, "labels.u8"))
    return {"file": "labels.u8", "index": [st["id"] for st in structures]}


def backfill_labels(case_dir):
    """Add labels.u8 to a case built before the overlay existed (uses the cached FastSurfer output)."""
    import nibabel as nib
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    seg_p = os.path.join(case_dir, "fastsurfer", "fs", "mri", "aparc.DKTatlas+aseg.deep.mgz")
    if not mf.get("structures") or not os.path.exists(seg_p):
        return False
    v = mf["volume"]
    lab = resample_labels(nib.load(seg_p), v["dims"], np.array(v["origin"]), float(v["spacing"][0]))
    manifest_io.update(case_dir, {"labelVolume": write_label_volume(case_dir, lab, mf["structures"])})
    return True


def build(case_dir, progress=lambda p, m: None, device="auto"):
    import nibabel as nib
    t0 = time.time()
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    vinfo = mf["volume"]
    nx, ny, nz = vinfo["dims"]
    origin = np.array(vinfo["origin"]); sp = float(vinfo["spacing"][0])
    vol = np.fromfile(os.path.join(case_dir, vinfo["file"]), np.uint8).reshape(nz, ny, nx)
    work = os.path.join(case_dir, "fastsurfer")
    os.makedirs(work, exist_ok=True)
    aff = np.diag([sp, sp, sp, 1.0]); aff[:3, 3] = origin
    t1 = os.path.join(work, "t1.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0)), aff), t1)
    progress(5, "Labelling structures with FastSurfer (on this computer)")
    seg, t_fs, _log = run_fastsurfer(t1, work, progress, device)
    timing = {"fastsurfer": t_fs}
    t = time.time()
    seg_np = np.asarray(seg.dataobj).astype(np.int16)
    vox_ml = float(abs(np.linalg.det(seg.affine[:3, :3]))) / 1000.0
    counts = dict(zip(*np.unique(seg_np, return_counts=True)))
    del seg_np
    lab = resample_labels(seg, (nx, ny, nz), origin, sp)
    timing["resample"] = round(time.time() - t, 1); t = time.time()
    sdir = os.path.join(case_dir, "structures"); os.makedirs(sdir, exist_ok=True)
    defs = STRUCT_DEFS["structures"]
    structures = []

    def add(key, hemi, mesh, labels_for_vol):
        sid = f"{key}_{hemi}" if hemi else key
        d = defs[key]
        fn = f"structures/{sid}.glb"
        export_glb(mesh, os.path.join(case_dir, fn), sid)
        vol_ml = sum(counts.get(l, 0) for l in labels_for_vol) * vox_ml
        structures.append({"id": sid, "key": key, "hemi": hemi, "group": d["group"],
                           "name": (HEMI_NAME[hemi] + " " + d["name"][0].lower() + d["name"][1:]).strip() if hemi else d["name"],
                           "color": d["color"], "file": fn, "faces": int(len(mesh.faces)),
                           "volume_ml": round(vol_ml, 1),
                           "centroid": [float(c) for c in mesh.centroid]})

    # ---- cortex lobes: split the Phase 1 surface by nearest cortical / other label
    progress(80, "Building cortex lobes")
    surf = trimesh.load(os.path.join(case_dir, "brain.glb"), force="mesh")
    cls = np.zeros_like(lab, np.int16)  # 1..12 lobes (hemi*6+lobe), 99 = hindbrain/other
    lobe_keys = ["frontal", "parietal", "temporal", "occipital", "cingulate", "insula"]
    parcel_index = {}  # (name, hemi) -> 1..N for parcels.u8
    parc = np.zeros_like(lab, np.uint8)
    for code, name in DKT.items():
        for base, hemi in ((1000, "L"), (2000, "R")):
            m = lab == base + code
            if not m.any():
                continue
            cls[m] = (0 if hemi == "L" else 6) + lobe_keys.index(LOBE[name]) + 1
            parcel_index.setdefault((name, hemi), len(parcel_index) + 1)
            parc[m] = parcel_index[(name, hemi)]
    other = np.isin(lab, [7, 8, 46, 47, 16, 28, 60, 15, 24])
    cls[other & (cls == 0)] = 99
    _, near = ndi.distance_transform_edt(cls == 0, return_indices=True)
    ijk = np.rint((surf.vertices - origin) / sp).astype(int)[:, ::-1]  # z,y,x
    ijk = np.clip(ijk, 0, np.array(cls.shape) - 1)
    nzi = near[:, ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    vlab = cls[nzi[0], nzi[1], nzi[2]]
    del near
    # majority vote over mesh neighbours so lobe borders follow the surface instead of voxel steps
    import scipy.sparse as sps
    e = surf.edges_unique
    A = sps.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                       shape=(len(vlab), len(vlab))).tocsr()
    A = A + sps.identity(len(vlab), format="csr")
    classes = np.unique(vlab)
    for _ in range(8):
        votes = np.stack([A @ (vlab == c).astype(np.float32) for c in classes], 1)
        vlab = classes[np.argmax(votes, 1)]
    flab = vlab[surf.faces]
    face_lab = np.where(flab[:, 1] == flab[:, 2], flab[:, 1], flab[:, 0])
    for li in range(1, 13):
        faces = np.where(face_lab == li)[0]
        if len(faces) < 50:
            continue
        sub = surf.submesh([faces], append=True)
        hemi = "L" if li <= 6 else "R"
        key = lobe_keys[(li - 1) % 6]
        labs = [b + c for c, n in DKT.items() if LOBE[n] == key for b in ((1000,) if hemi == "L" else (2000,))]
        add(key, hemi, sub, labs)
    # cortex that is not on the outer surface (cingulate on the medial wall, insula under the
    # opercula): closed meshes from the cortical ribbon labels instead
    have = {(st["key"], st["hemi"]) for st in structures}
    for key in lobe_keys:
        for hemi, base in (("L", 1000), ("R", 2000)):
            if (key, hemi) in have:
                continue
            labs = [base + c for c, n in DKT.items() if LOBE[n] == key]
            m = np.isin(lab, labs)
            mesh = mask_to_mesh(m, origin, sp, sigma=0.8, target_faces=30_000, smooth_iters=10,
                                min_component_frac=0.05) if m.sum() > 200 else None
            if mesh is not None:
                add(key, hemi, mesh, labs)
    # parcels.u8 for click lookup: nearest gyrus within ~4 mm of cortex
    d, near = ndi.distance_transform_edt(parc == 0, return_indices=True)
    pfill = parc[near[0], near[1], near[2]]
    pfill[d > 4] = 0
    pfill.astype(np.uint8).tofile(os.path.join(case_dir, "parcels.u8"))
    del d, near, pfill
    timing["cortex"] = round(time.time() - t, 1); t = time.time()

    # ---- closed meshes from labels
    keys = list(ASEG)
    for i, key in enumerate(keys):
        progress(85 + int(12 * i / len(keys)), f"Building {defs[key]['name']}")
        left, right = ASEG[key]
        for hemi, labels in (("L", left), ("R", right)) if right else ((None, left),):
            m = np.isin(lab, labels)
            if key == "cerebral_wm":  # include WM hypointensities on the matching side
                wmh = lab == 77
                xs = np.arange(nx)[None, None, :] * sp + origin[0]
                mid = mf.get("brainCenter", [0, 0, 0])[0]
                m |= wmh & ((xs < mid) if hemi == "L" else (xs >= mid))
            if m.sum() < 30:
                continue
            mesh = mask_to_mesh(m, origin, sp, **MESH_OPTS.get(key, DEFAULT_OPTS))
            if mesh is not None:
                add(key, hemi, mesh, labels)
    timing["meshes"] = round(time.time() - t, 1)
    timing["total"] = round(time.time() - t0, 1)
    info = {"tool": "FastSurfer v2.6.0 (FastSurferVINN)", "device": device, "timing": timing,
            "totalSegVolumeMl": round(sum(c for l, c in counts.items() if l not in (0, 24)) * vox_ml, 1)}
    # only the keys Phase 2 owns: other writers (scan alignment, veins) update the same file meanwhile
    manifest_io.update(case_dir, {
        "labelVolume": write_label_volume(case_dir, lab, structures), "phase": 2, "structures": structures,
        "structureGroups": STRUCT_DEFS["groups"], "structureInfo": {k: v["description"] for k, v in defs.items()},
        "parcels": {"file": "parcels.u8", "index": {str(i): [n, h] for (n, h), i in parcel_index.items()},
                    "info": STRUCT_DEFS["parcels"]},
        "phase2": info})
    progress(100, "Structures ready")
    return info

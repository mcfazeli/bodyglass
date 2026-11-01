"""End-to-end Phase 1 pipeline: DICOM folder -> case directory for the viewer."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

from .dicom_scan import pick_t1, scan_folder
from .layers import LAYERS
from .meshes import export_glb, mask_to_mesh
from . import airclean, manifest_io, sequences, synthstrip
from .segment import SKIN_VERSION, head_mask, seal_face, largest_cc, normalize, segment
from .volume import load_series, to_ras_grid

MESH_PARAMS = {
    "skin": dict(sigma=1.5, target_faces=160_000, smooth_iters=10),
    "skull": dict(sigma=2.0, target_faces=80_000, smooth_iters=10),
    "brain": dict(sigma=0.7, target_faces=400_000, smooth_iters=12, min_component_frac=0.05),
    "white_matter": dict(sigma=0.7, target_faces=300_000, smooth_iters=12, min_component_frac=0.05),
    "ventricles": dict(sigma=1.0, target_faces=30_000, smooth_iters=10, min_component_frac=0.1),
}


def case_id_for(series_uid: str) -> str:
    return hashlib.sha1(series_uid.encode()).hexdigest()[:12]


def run(folder: str, cases_root: str, progress=lambda pct, msg: None, uid: str | None = None) -> str:
    """uid: build from this series instead of the auto-picked T1 ('Use for 3D instead')."""
    t0 = time.time()
    timing = {}

    def lap(name, t):
        timing[name] = round(time.time() - t, 2)
        return time.time()

    progress(2, "Reading DICOM headers")
    t = time.time()
    series = scan_folder(folder)
    t = lap("scan_headers", t)
    best = pick_t1(series)
    if uid:
        best = next((s for s in series if s.uid == uid), None)
        if best is None:
            raise RuntimeError("That series is no longer in the scan folder.")
    if best is None:
        raise RuntimeError("No 3D T1-weighted series found in this folder.")
    cid = case_id_for(best.uid)
    out = os.path.join(cases_root, cid)
    if os.path.exists(os.path.join(out, "manifest.json")):
        # shortcut: a finished case is reused as is (older ones are upgraded by the server's backfills);
        # delete the case folder to rebuild it from scratch.
        progress(100, "Already processed on this computer")
        return cid
    os.makedirs(out, exist_ok=True)
    try:   # which other scans (FLAIR, SWI, DWI...) exist; they are aligned later in the background
        seq_picks = sequences.write_series_json(out, series, best)
    except Exception as e:
        seq_picks = {}
        print("sequence detection failed:", e)

    progress(10, f"Loading {best.meta.get('SeriesDescription')} ({len(best.files)} images)")
    vol, aff = load_series(best.files)
    t = lap("load_pixels", t)
    progress(20, "Resampling to 1 mm grid")
    grid, origin, sp = to_ras_grid(vol, aff, spacing=1.0)
    del vol
    v = normalize(grid)
    del grid
    t = lap("resample", t)

    strip_info = {"method": "classic"}
    brain_mask = None
    if os.environ.get("BODYGLASS_STRIP", "synthstrip") == "synthstrip" and synthstrip.available():
        progress(25, "Finding the brain (SynthStrip, on this computer)")
        try:
            brain_mask, strip_info = synthstrip.run(v)
            strip_info["method"] = "synthstrip"
        except Exception as e:  # fall back to the classic estimate
            strip_info = {"method": "classic", "synthstripError": str(e)}
        t = lap("synthstrip", t)
    progress(35, "Outlining skin, brain, white matter and ventricles")
    seg = segment(v, brain_mask)
    del brain_mask
    t = lap("segment", t)
    # viewer volume (uint8, x fastest). Air outside the head is set to 0 so scanner noise (strong on
    # MP2RAGE scans) does not show as speckle; airclean below removes what is left near the scalp.
    v8 = (v * 255).astype(np.uint8)
    head = seg["head"]
    if head.mean() > 0.05:
        keep = ndi.binary_dilation(head, iterations=3)
        # salt-and-pepper background: a 5 mm median filter removes isolated bright voxels
        med = largest_cc(ndi.median_filter(v8, size=5) > 30)
        med = np.stack([ndi.binary_fill_holes(sl) for sl in ndi.binary_dilation(med, iterations=2)])
        if med.mean() > 0.05:
            keep &= med
        v8[~keep] = 0
    # brain zone (brain + 4 mm) so colour overlays of other scans tint the brain, not the scalp
    zone = ndi.binary_dilation(seg["brain_env"], iterations=4)
    zone.astype(np.uint8).tofile(os.path.join(out, "brainzone.u8"))
    airclean.clean(v8, zone).tofile(os.path.join(out, "volume.u8"))
    del v8, zone
    masks = {
        "skin": seal_face(seg["head"], ndi.center_of_mass(seg["brain_env"])),
        "skull": seg["skull"],
        "brain": ndi.binary_fill_holes(seg["brain"]),
        "white_matter": seg["white_matter"],
        "ventricles": seg["ventricles"],
    }
    # Sub-voxel surfaces for brain + white matter: iso-surface of the image intensity itself,
    # restricted to the (slightly grown) masks; internal dark holes are filled first.
    sm, st = seg["smooth"], seg["stats"]
    deep = ndi.distance_transform_edt(seg["brain_env"]) > 6
    wm_big = st["wm_peak"]
    # open (erode 2 mm, keep main part, re-dilate) to drop thin sheets: vessels, dura, optic nerves
    core = largest_cc(ndi.distance_transform_edt(masks["brain"]) > 2.0)
    region = (ndi.distance_transform_edt(~core) <= 3.5) & seg["brain_env"]
    f_brain = np.where(region, sm, 0).astype(np.float32)
    f_brain[deep & (f_brain < wm_big)] = wm_big
    wm_region = ndi.distance_transform_edt(~seg["white_matter"]) <= 1.5
    inside_brain = ndi.distance_transform_edt(masks["brain"]) > 1.5  # keep WM under the cortex surface
    f_wm = np.where(wm_region & seg["brain_env"] & inside_brain, sm, 0).astype(np.float32)
    fields = {"brain": (f_brain, st["t_brain"]), "white_matter": (f_wm, st["t_wm"])}
    # "sulcal depth" shading: distance inside the closed brain envelope, sampled per vertex
    depth_map = ndi.distance_transform_edt(seg["brain_env"]).astype(np.float32) * sp
    layers = []
    for i, L in enumerate(LAYERS):
        progress(45 + 10 * i, f"Building 3D surface: {L['name']}")
        if L["id"] in fields:
            fld, lvl = fields[L["id"]]
            m = mask_to_mesh(fld, origin, sp, level=lvl, **MESH_PARAMS[L["id"]])
        else:
            m = mask_to_mesh(masks[L["id"]], origin, sp, **MESH_PARAMS[L["id"]])
        if m is None:
            continue
        if L["id"] in ("brain", "white_matter"):
            ijk = ((m.vertices - origin) / sp)[:, ::-1].T  # z,y,x
            d = ndi.map_coordinates(depth_map, ijk, order=1)
            ref = 0.3 if L["id"] == "brain" else 3.0
            k = np.clip((d - ref) / 5.0, 0, 1) ** 0.6
            deep_col = np.array([0.50, 0.30, 0.33]) if L["id"] == "brain" else np.array([0.62, 0.55, 0.50])
            rgb = 1.0 - k[:, None] * (1.0 - deep_col[None, :])
            c = (np.clip(rgb, 0, 1) ** 2.2 * 255).astype(np.uint8)  # glTF vertex colours are linear
            m.visual.vertex_colors = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], 1)
        fn = f"{L['id']}.glb"
        size = export_glb(m, os.path.join(out, fn), L["id"])
        layers.append({**L, "file": fn, "faces": int(len(m.faces)), "bytes": size,
                       "volume_ml": round(float(masks[L["id"]].sum()) * sp ** 3 / 1000, 1)})
        t = lap(f"mesh_{L['id']}", t)
    timing["total"] = round(time.time() - t0, 2)
    nz, ny, nx = v.shape
    manifest = {
        "caseId": cid,
        "phase": 1,
        "sourceFolder": folder,
        "studyDate": best.meta.get("StudyDate"),   # for 'Use for 3D instead' (rebuild from another series of the same exam)
        "series": {"chosen": best.summary(), "all": [dict(s.summary(), kind=s.meta.get("kind")) for s in series]},
        "sequencesFound": {k: p["series"].meta.get("SeriesDescription", "") for k, p in seq_picks.items()},
        "volume": {"file": "volume.u8", "dims": [nx, ny, nz], "spacing": [sp] * 3,
                   "origin": [float(x) for x in origin], "space": "RAS"},
        "brainCenter": [float(c) for c in (np.array(ndi.center_of_mass(masks["brain"]))[::-1] * sp + origin)],
        "layers": layers,
        "brainZone": "brainzone.u8",
        "airClean": airclean.VERSION,
        "skinVersion": SKIN_VERSION,
        "brainExtraction": strip_info,
        "volumesMl": {
            "intracranial_mask": round(float(seg["brain_env"].sum()) * sp ** 3 / 1000, 1),
            "brain_tissue": round(float((seg["brain_env"] & (seg["smooth"] > seg["stats"]["t_brain"])).sum()) * sp ** 3 / 1000, 1),
            "white_matter": round(float(seg["white_matter"].sum()) * sp ** 3 / 1000, 1),
        },
        "timing": timing,
        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_io.write(out, manifest)
    progress(100, "Done")
    return cid


def reskin(case_dir):
    """Rebuild only the skin surface of an existing case with the current seal_face (offered once in the UI);
    the brain, structures and slices are untouched, nothing is re-read from the scan folder."""
    from . import manifest_io
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    nx, ny, nz = mf["volume"]["dims"]
    origin, sp = np.array(mf["volume"]["origin"]), mf["volume"]["spacing"][0]
    v = np.fromfile(os.path.join(case_dir, mf["volume"]["file"]), np.uint8).reshape(nz, ny, nx).astype(np.float32) / 255
    head = head_mask(ndi.gaussian_filter(v, 1.0))
    del v
    center = ((np.array(mf["brainCenter"]) - origin) / sp)[::-1]
    m = mask_to_mesh(seal_face(head, center), origin, sp, **MESH_PARAMS["skin"])
    layers = mf["layers"]
    for L in layers:
        if L["id"] == "skin":
            L.update(faces=int(len(m.faces)), bytes=export_glb(m, os.path.join(case_dir, L["file"]), "skin"))
    manifest_io.update(case_dir, {"layers": layers, "skinVersion": SKIN_VERSION})

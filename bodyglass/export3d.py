"""3D-printable export (STL / 3MF) of chosen layers or structures, in millimetres.

The chosen meshes are rasterised into one solid voxel grid (surface samples + hole filling per
part), merged, and re-meshed with marching cubes. This always gives a closed (watertight),
manifold surface even where structures touch or overlap, which slicers need. Coordinates are
the scan's own RAS millimetres, so the print is life size (1 unit = 1 mm)."""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure


def _sources(case_dir, mf, ids):
    """ids: layer ids, 'g:<group>', structure ids, or 'veins'. Returns [(name, path)]."""
    out = []
    for L in mf.get("layers", []):
        if L["id"] in ids:
            out.append((L["name"], L["file"]))
    for st in mf.get("structures", []) + (mf.get("neck") or {}).get("structures", []):
        if st["id"] in ids or ("g:" + st["group"]) in ids:
            out.append((st["name"], st["file"]))
    if "veins" in ids and (mf.get("veins") or {}).get("file"):
        out.append(("Veins", mf["veins"]["file"]))
    return [(n, os.path.join(case_dir, f)) for n, f in out if os.path.exists(os.path.join(case_dir, f))]


def _load(path):
    g = trimesh.load(path, force="scene")
    return trimesh.util.concatenate([m for m in g.dump() if isinstance(m, trimesh.Trimesh)])


def _fill3(part):
    """Solid from a closed-ish shell: 3-D cavity fill, or a vote of 2-D fills along the three axes
    (robust to small gaps where touching surfaces were cut)."""
    full = ndi.binary_fill_holes(part)
    if full.sum() > part.sum() * 1.5:
        return full
    votes = sum(np.moveaxis(np.stack([ndi.binary_fill_holes(sl) for sl in np.moveaxis(part, ax, 0)]).astype(np.uint8), 0, ax)
                for ax in range(3))
    return part | (votes >= 2)


def build_solid(meshes, pitch=None, max_faces=500_000, masks=()):
    """meshes: trimesh parts; masks: (bool array z,y,x, origin RAS mm, spacing mm) from the label volume."""
    lows = [m.bounds[0] for m in meshes] + [o + np.argwhere(k).min(0)[::-1] * sp for k, o, sp in masks]
    highs = [m.bounds[1] for m in meshes] + [o + np.argwhere(k).max(0)[::-1] * sp for k, o, sp in masks]
    lo = np.min(lows, 0) - 3
    hi = np.max(highs, 0) + 3
    ext = hi - lo
    pitch = pitch or float(max(0.6, ext.max() / 300))
    shape = np.ceil(ext / pitch).astype(int) + 1
    solid = np.zeros(shape[::-1], bool)                       # z, y, x
    for k, o, sp in masks:   # labelled structures: resample the 1 mm labels onto the print grid (smooth)
        zz, yy, xx = [np.arange(n) * pitch + lo[a] for n, a in zip(shape[::-1], (2, 1, 0))]
        crop = ndi.gaussian_filter(k.astype(np.float32), 0.6)
        for zi, z in enumerate(zz):   # slice by slice (memory)
            kz = (z - o[2]) / sp
            if kz < -1 or kz > k.shape[0]:
                continue
            Y, X = np.meshgrid((yy - o[1]) / sp, (xx - o[0]) / sp, indexing="ij")
            val = ndi.map_coordinates(crop, [np.full(Y.shape, kz), Y, X], order=1, cval=0.0)
            solid[zi] |= val > 0.5
    for m in meshes:   # each part in its own bounding box (memory), filled solid, then merged
        vi = np.round((m.vertices - lo) / pitch).astype(np.int32)
        a0 = np.maximum(vi.min(0) - 2, 0); a1 = np.minimum(vi.max(0) + 3, shape)
        part = np.zeros((a1 - a0)[::-1], bool)
        total = int(m.area / (pitch * 0.45) ** 2)              # dense enough that the shell has no gaps
        for k in range(0, total, 2_000_000):                     # in chunks (memory)
            pts, _ = trimesh.sample.sample_surface(m, min(2_000_000, total - k), seed=k)
            ijk = np.round((pts - lo) / pitch).astype(np.int32) - a0
            ijk = np.clip(ijk, 0, (a1 - a0) - 1)
            part[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True
        vi -= a0; part[vi[:, 2], vi[:, 1], vi[:, 0]] = True
        solid[a0[2]:a1[2], a0[1]:a1[1], a0[0]:a1[0]] |= _fill3(ndi.binary_closing(part, iterations=1))
    solid = ndi.binary_fill_holes(solid)
    # drop specks so the print has no loose crumbs
    lab, n = ndi.label(solid)
    if n > 1:
        sizes = np.bincount(lab.ravel()); keep = sizes >= max(50, 0.0005 * sizes[1:].sum()); keep[0] = False
        solid = keep[lab]
    f = ndi.gaussian_filter(np.pad(solid, 2).astype(np.float32), 0.7); del solid
    v, faces, _, _ = measure.marching_cubes(f, 0.5)
    v = (v[:, ::-1] - 2) * pitch + lo
    mesh = trimesh.Trimesh(v, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    if len(mesh.faces) > max_faces:
        import fast_simplification
        vv, ff = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                              target_reduction=1 - max_faces / len(mesh.faces))
        simp = trimesh.Trimesh(vv, ff, process=True)
        simp.merge_vertices(); simp.update_faces(simp.nondegenerate_faces()); simp.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(simp)
        if simp.is_watertight:
            mesh = simp
        else:   # fall back to a coarser marching-cubes step, which is always closed
            step = int(np.ceil(np.sqrt(len(mesh.faces) / max_faces)))
            v, faces, _, _ = measure.marching_cubes(f, 0.5, step_size=step)
            mesh = trimesh.Trimesh((v[:, ::-1] - 2) * pitch + lo, faces, process=True)
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    parts = mesh.split(only_watertight=True)
    if len(parts) > 1:   # crumbs from the coarse fallback step: keep real parts only
        big = max(len(q.faces) for q in parts)
        mesh = trimesh.util.concatenate([q for q in parts if len(q.faces) >= min(max(300, 0.002 * big), big)])
    mesh.fix_normals()
    return mesh, pitch


def write_3mf(mesh, name):
    """Minimal 3MF (core spec): one object, unit = millimeter. No extra dependencies."""
    import zipfile
    v = "".join(f'<vertex x="{x:.3f}" y="{y:.3f}" z="{z:.3f}"/>' for x, y, z in mesh.vertices)
    t = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<metadata name="Title">{name}</metadata><metadata name="Application">BodyGlass</metadata>'
             f'<resources><object id="1" type="model" name="{name}"><mesh><vertices>{v}</vertices><triangles>{t}</triangles></mesh></object></resources>'
             '<build><item objectid="1"/></build></model>')
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels); z.writestr("3D/3dmodel.model", model)
    return out.getvalue()


def export(case_dir, ids, fmt="stl"):
    t0 = time.time()
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        mf = json.load(fh)
    ids = set(ids)
    src = _sources(case_dir, mf, ids)
    masks, names = [], []
    LV = mf.get("labelVolume")
    if LV and os.path.exists(os.path.join(case_dir, LV["file"])):   # structures: use the solid labels
        want = [i + 1 for i, sid in enumerate(LV["index"])
                if sid in ids or any(st["id"] == sid and ("g:" + st["group"]) in ids for st in mf.get("structures", []))]
        if want:
            nx, ny, nz = mf["volume"]["dims"]
            lab = np.fromfile(os.path.join(case_dir, LV["file"]), np.uint8).reshape(nz, ny, nx)
            k = np.isin(lab, want)
            if k.any():
                masks.append((k, np.asarray(mf["volume"]["origin"], float), float(mf["volume"]["spacing"][0])))
                st_names = {st["file"] for st in mf.get("structures", [])}
                names = [n for n, p in src if os.path.relpath(p, case_dir) in st_names]
                src = [(n, p) for n, p in src if os.path.relpath(p, case_dir) not in st_names]
    if not src and not masks:
        raise ValueError("Nothing selected to export.")
    meshes = [_load(p) for _, p in src]
    mesh, pitch = build_solid(meshes, masks=masks)
    src = [(n, None) for n in names] + src
    parts = len(mesh.split(only_watertight=False))
    buf = io.BytesIO()
    if fmt == "3mf":
        buf.write(write_3mf(mesh, "BodyGlass model"))
    else:
        buf.write(trimesh.exchange.stl.export_stl(mesh))
    size = (mesh.bounds[1] - mesh.bounds[0])
    info = {"parts": [n for n, _ in src], "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
            "bodies": int(parts), "sizeMm": [round(float(x), 1) for x in size], "volumeMl": round(float(mesh.volume) / 1000, 1),
            "voxelMm": round(pitch, 2), "seconds": round(time.time() - t0, 1), "format": fmt}
    return buf.getvalue(), info

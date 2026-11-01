"""Masks -> smoothed, decimated triangle meshes in RAS millimetres -> GLB."""
from __future__ import annotations

import fast_simplification
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure


def mask_to_mesh(mask, origin, spacing, sigma=1.0, target_faces=200_000,
                 smooth_iters=10, step=1, level=0.5, min_component_frac=0.0):
    """mask: bool mask (iso-surface at 0.5 after blur) or a float field with its own level."""
    if mask.dtype == bool and mask.sum() < 50:
        return None
    pad = 3
    m = np.pad(mask.astype(np.float32), pad)
    f = ndi.gaussian_filter(m, sigma) if sigma > 0 else m
    verts, faces, _n, _v = measure.marching_cubes(f, level=level, step_size=step,
                                                  allow_degenerate=False)
    # (z,y,x) voxel -> RAS mm
    verts = (verts[:, ::-1] - pad) * spacing + np.asarray(origin)[None, :]
    if len(faces) > target_faces:
        verts, faces = fast_simplification.simplify(
            verts.astype(np.float32), faces.astype(np.int32),
            target_reduction=1 - target_faces / len(faces))
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if min_component_frac > 0:  # drop small floating bits (vessels, dura flecks)
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            big = max(len(p.faces) for p in parts)
            mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= min_component_frac * big])
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    mesh.fix_normals()  # consistent outward winding
    return mesh


def export_glb(mesh, path, name):
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    data = scene.export(file_type="glb")
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)

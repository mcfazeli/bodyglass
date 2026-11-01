"""Load a DICOM series into a RAS-aligned isotropic grid."""
from __future__ import annotations

import numpy as np
import pydicom
from scipy import ndimage


def load_series(files: list[str]):
    """Return (data[k,j,i] float32, affine 4x4 voxel(i,j,k)->LPS mm)."""
    dss = [pydicom.dcmread(f) for f in files]
    dss = [d for d in dss if hasattr(d, "ImagePositionPatient")]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    row_dir, col_dir = iop[:3], iop[3:]  # row_dir: +column index, col_dir: +row index
    normal = np.cross(row_dir, col_dir)
    dss.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    # drop duplicate positions (e.g. multi-echo); keep first
    seen, uniq = set(), []
    for d in dss:
        key = round(float(np.dot(normal, np.array(d.ImagePositionPatient, float))), 3)
        if key not in seen:
            seen.add(key); uniq.append(d)
    dss = uniq
    ps = [float(x) for x in dss[0].PixelSpacing]  # [row spacing, col spacing]
    p0 = np.array(dss[0].ImagePositionPatient, float)
    if len(dss) > 1:
        step = (np.array(dss[-1].ImagePositionPatient, float) - p0) / (len(dss) - 1)
    else:
        step = normal * float(getattr(dss[0], "SliceThickness", 1.0))
    aff = np.eye(4)
    aff[:3, 0] = row_dir * ps[1]
    aff[:3, 1] = col_dir * ps[0]
    aff[:3, 2] = step
    aff[:3, 3] = p0
    vol = np.stack([
        d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1)
        + float(getattr(d, "RescaleIntercept", 0) or 0)
        for d in dss
    ])  # (k, rows=j, cols=i)
    return vol, aff


def to_ras_grid(vol, aff_lps, spacing=1.0):
    """Resample to an axis-aligned RAS grid. Returns (grid[z,y,x], origin_ras, spacing)."""
    lps2ras = np.diag([-1, -1, 1, 1.0])
    aff = lps2ras @ aff_lps
    nk, nj, ni = vol.shape
    corners = np.array([[i, j, k, 1] for i in (0, ni - 1) for j in (0, nj - 1) for k in (0, nk - 1)]).T
    w = (aff @ corners)[:3]
    lo, hi = w.min(1), w.max(1)
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1  # x,y,z
    inv = np.linalg.inv(aff)
    out = np.zeros((shape[2], shape[1], shape[0]), np.float32)
    xs = lo[0] + np.arange(shape[0]) * spacing
    ys = lo[1] + np.arange(shape[1]) * spacing
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # (ny,nx)
    for zi in range(shape[2]):
        z = lo[2] + zi * spacing
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z), np.ones(X.size)])
        ijk = (inv @ pts)[:3]
        coords = np.stack([ijk[2], ijk[1], ijk[0]])  # k, j, i
        out[zi] = ndimage.map_coordinates(vol, coords, order=1, cval=0.0,
                                          prefilter=False).reshape(X.shape)
    return out, lo, spacing

"""Load a DICOM series into a RAS-aligned isotropic grid."""
from __future__ import annotations

import numpy as np
import pydicom
from scipy import ndimage

from .dicom_scan import BAD_FRAMES, frame_attr, frames_bad, split_ref, unsupported_syntax


def _slices(files):
    """(position, orientation, pixel spacing, thickness, pixels) per slice. A file path is one classic slice;
    a "<path>#frame=<n>" reference is one frame of an enhanced multi-frame file, read once."""
    out, cache = [], {}
    for ref in files:
        path, f = split_ref(ref)
        if path not in cache:
            cache[path] = pydicom.dcmread(path)
            if f is not None and frames_bad(cache[path], path):
                raise RuntimeError(BAD_FRAMES)
        d = cache[path]
        at = lambda kw: frame_attr(d, f, kw)
        if at("ImagePositionPatient") is None or at("ImageOrientationPatient") is None or at("PixelSpacing") is None:
            if f is None:
                continue
            raise RuntimeError("This multi-frame DICOM file has no slice positions, so BodyGlass can't build "
                               "a 3D model from it. Try a classic (one file per slice) export of the scan.")
        if msg := unsupported_syntax(d):
            raise RuntimeError(msg)
        try:
            px = d.pixel_array
        except Exception as e:   # a compressed claim that doesn't decode: plain message, not a pydicom error
            if f is None:
                raise
            raise RuntimeError(BAD_FRAMES) from e
        px = px[f] if f is not None and px.ndim == 3 else px
        out.append((np.array(at("ImagePositionPatient"), float), np.array(at("ImageOrientationPatient"), float),
                    [float(x) for x in at("PixelSpacing")], float(at("SliceThickness") or 1.0),
                    px.astype(np.float32) * float(at("RescaleSlope") or 1) + float(at("RescaleIntercept") or 0)))
    return out


def load_series(files: list[str]):
    """Return (data[k,j,i] float32, affine 4x4 voxel(i,j,k)->LPS mm)."""
    sl = _slices(files)
    iop = sl[0][1]
    row_dir, col_dir = iop[:3], iop[3:]  # row_dir: +column index, col_dir: +row index
    normal = np.cross(row_dir, col_dir)
    sl.sort(key=lambda d: float(np.dot(normal, d[0])))
    # drop duplicate positions (e.g. multi-echo); keep first
    seen, uniq = set(), []
    for d in sl:
        key = round(float(np.dot(normal, d[0])), 3)
        if key not in seen:
            seen.add(key); uniq.append(d)
    sl = uniq
    ps = sl[0][2]  # [row spacing, col spacing]
    p0 = sl[0][0]
    step = (sl[-1][0] - p0) / (len(sl) - 1) if len(sl) > 1 else normal * sl[0][3]
    aff = np.eye(4)
    aff[:3, 0] = row_dir * ps[1]
    aff[:3, 1] = col_dir * ps[0]
    aff[:3, 2] = step
    aff[:3, 3] = p0
    vol = np.stack([d[4] for d in sl])  # (k, rows=j, cols=i)
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

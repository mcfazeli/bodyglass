"""Remove scanner background noise (speckle) around the head from the viewer's T1 volume.

MP2RAGE scans have dense salt-and-pepper noise in the air. Real tissue is locally smooth (low
local standard deviation relative to its brightness); noise is not. Keep smooth bright regions
within 18 mm of the brain, plus the brain zone itself, fill holes. Takes ~3 s for a 256^3 volume."""
import numpy as np
from scipy import ndimage as ndi

from .segment import largest_cc

VERSION = 2


def clean(v8: np.ndarray, zone: np.ndarray) -> np.ndarray:
    f = v8.astype(np.float32)
    mu = ndi.uniform_filter(f, 3)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(f * f, 3) - mu * mu, 0))
    smooth = ndi.binary_opening((sd < 0.3 * mu + 4) & (mu > 20), iterations=1)
    near = ndi.distance_transform_edt(~zone) <= 18
    mk = (smooth & near) | zone
    mk = ndi.binary_closing(largest_cc(mk), iterations=2)
    mk = np.stack([ndi.binary_fill_holes(s) for s in mk])
    out = v8.copy()
    out[~mk] = 0
    return out


def backfill(case_dir):
    """Older cases: clean volume.u8 in place once (needs brainzone.u8)."""
    import json, os
    from . import manifest_io
    with open(os.path.join(case_dir, "manifest.json")) as fh:
        m = json.load(fh)
    zp = os.path.join(case_dir, m.get("brainZone") or "brainzone.u8")
    if m.get("airClean", 0) >= VERSION or not os.path.exists(zp):
        return False
    nx, ny, nz = m["volume"]["dims"]
    vp = os.path.join(case_dir, m["volume"]["file"])
    v = np.fromfile(vp, np.uint8).reshape(nz, ny, nx)
    z = np.fromfile(zp, np.uint8).reshape(nz, ny, nx) > 0
    clean(v, z).tofile(vp + ".tmp")
    os.replace(vp + ".tmp", vp)
    manifest_io.update(case_dir, {"airClean": VERSION})
    return True

"""Phase 1 quick segmentation (classical, CPU, seconds).

Produces approximate masks for: head (skin), skull (estimated shell), brain,
white matter and lateral ventricles. Phase 2 replaces brain/deep structures
with a learned segmentation (FastSurfer / SynthSeg) - see ARCHITECTURE.md.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology


def _ball(r):
    return morphology.ball(r).astype(bool)


def dilate(m, r):
    """Euclidean dilation by r voxels (distance transform; much faster than big structuring elements)."""
    return ndi.distance_transform_edt(~m) <= r


def erode(m, r):
    return ndi.distance_transform_edt(m) > r


def close(m, r, pad=None):
    pad = int(np.ceil(r)) + 1 if pad is None else pad
    mp = np.pad(m, pad)
    return erode(dilate(mp, r), r)[pad:-pad, pad:-pad, pad:-pad]


def largest_cc(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def normalize(vol):
    lo, hi = np.percentile(vol[vol > 0], [0.5, 99.5]) if (vol > 0).any() else (0, 1)
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def estimate_wm_peak(sm, head):
    """WM intensity from a box in the middle of the cranium (top of head - 55 mm)."""
    zs = np.where(head.any(axis=(1, 2)))[0]
    zc = max(int(zs.max()) - 55, int(zs.min()))
    ys, xs = np.nonzero(head[zc])
    yc, xc = int(ys.mean()), int(xs.mean())
    box = sm[max(zc - 20, 0):zc + 25, max(yc - 25, 0):yc + 25, max(xc - 25, 0):xc + 25]
    vals = box[box > np.median(box)]
    hist, edges = np.histogram(vals, bins=100, range=(0, 1))
    hs = ndi.gaussian_filter1d(hist.astype(float), 2)
    return float((edges[:-1] + edges[1:])[np.argmax(hs)] / 2)


def pick_ventricles(csf, env):
    """Lateral ventricles = dark blobs near the brain centre that do not reach the top of the
    brain (which rejects the interhemispheric fissure and sulcal CSF)."""
    lab, n = ndi.label(csf)
    vent = np.zeros_like(csf)
    if not n:
        return vent
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(csf, lab, idx)
    cents = np.array(ndi.center_of_mass(csf, lab, idx))
    boxes = ndi.find_objects(lab)
    c0 = np.array(ndi.center_of_mass(env))
    ztop = np.where(env.any(axis=(1, 2)))[0].max()
    dist = np.linalg.norm((cents - c0) * np.array([1.0, 0.8, 1.6]), axis=1)  # z,y,x weights
    zmax = np.array([b[0].stop for b in boxes])
    ok = (dist < 40) & (sizes > 300) & (zmax < ztop - 20)
    if ok.any():
        best = sizes[ok].max()
        for i in np.where(ok)[0]:
            if sizes[i] > 0.25 * best:
                vent |= lab == idx[i]
    return vent


def head_mask(sm):
    """Head (skin surface) mask from the smoothed volume."""
    t_bg = filters.threshold_otsu(sm[::2, ::2, ::2])
    head = largest_cc(sm > t_bg * 0.5)
    head = close(head, 3)
    head = ndi.binary_fill_holes(head)
    return np.stack([ndi.binary_fill_holes(sl) for sl in head])  # axial: closes nostrils/ears


SKIN_VERSION = 2   # 2: mouth / nostril openings sealed (seal_face)


def seal_face(head, center, r=10, half_width=35):
    """Teeth and air give no MRI signal, so an open mouth or nostril shows as a hole in the skin.
    Bridge openings up to ~2 r mm wide, only in front of and below the brain centre and near the midline
    (center = z, y, x voxel of the brain centre; 1 mm grid, y grows anterior). Back of head, ears and brain untouched.
    shortcut: the closing also rounds the chin-neck corner slightly (radius < r)."""
    closed = close(head, r)
    closed = np.stack([ndi.binary_fill_holes(sl) for sl in closed])
    z, y, x = np.ogrid[:head.shape[0], :head.shape[1], :head.shape[2]]
    zone = (z < center[0]) & (y > center[1]) & (np.abs(x - center[2]) < half_width)
    return head | (closed & zone)


def segment(v, brain_mask=None):
    """v: normalized float volume [z,y,x] at ~1 mm. Returns dict of masks + stats.

    brain_mask: optional learned brain mask (SynthStrip, includes CSF around the brain).
    Without it the classic threshold/erosion estimate is used (fallback)."""
    sm = ndi.gaussian_filter(v, 1.0)
    # --- head / skin
    head = head_mask(sm)
    if brain_mask is not None:
        env = brain_mask & head if (brain_mask & head).sum() > 0.9 * brain_mask.sum() else brain_mask
        # 3-class Otsu inside the learned mask: CSF | grey matter | white matter (contrast-adaptive)
        vals = sm[env][::7]
        t_csf_gm, t_gm_wm = filters.threshold_multiotsu(vals, classes=3)
        wm = float(np.median(vals[vals > t_gm_wm]))
        t_brain, t_wm = float(t_csf_gm), float(t_gm_wm)
        brain = largest_cc(env & (sm > t_brain))
        method = "synthstrip"
    else:
        wm = estimate_wm_peak(sm, head)
        # GM roughly 0.65-0.75 of WM on MPRAGE; CSF much darker
        t_brain = 0.50 * wm
        t_wm = 0.86 * wm
        # --- brain: threshold, erode to cut bridges to scalp/eyes, keep the big central blob
        cand = (sm > t_brain) & (sm < 1.6 * wm) & erode(head, 4)
        er = erode(cand, 6.0)
        brain = dilate(largest_cc(er), 6.5) & cand
        brain = largest_cc(brain)
        # solid brain envelope (for skull / ventricles): close sulci and fill
        env = close(brain, 6)
        env = ndi.binary_fill_holes(env)
        env = largest_cc(ndi.binary_opening(env, _ball(2)))
        method = "classic"
    # --- white matter
    wmm = largest_cc((sm > t_wm) & env)
    # --- lateral ventricles: dark CSF deep inside the envelope
    deep = erode(env, 8 if method == "synthstrip" else 6)
    if method == "classic":
        csf = ndi.binary_opening((sm < 0.42 * wm) & deep, _ball(1))
    else:
        csf = (sm < 0.95 * t_brain) & deep  # below the CSF/grey-matter Otsu threshold
    vent = pick_ventricles(ndi.binary_opening(csf, _ball(1)), env)
    if not vent.any():
        vent = pick_ventricles(csf, env)
    # --- skull (approximate shell between brain envelope and scalp)
    skull = dilate(env, 7.5) & erode(head, 3)
    return {
        "head": head, "skull": skull, "brain_env": env, "brain": brain,
        "white_matter": wmm, "ventricles": vent,
        "stats": {"wm_peak": wm, "t_brain": t_brain, "t_wm": t_wm, "method": method},
        "smooth": sm,
    }

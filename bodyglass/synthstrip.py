"""SynthStrip brain extraction, standalone, local, CPU or Apple-Silicon MPS.

Model: Hoopes et al., "SynthStrip: Skull-Stripping for Any Brain Image", NeuroImage 2022.
Weights `synthstrip.1.pt` (29 MB) are distributed by the FreeSurfer team under MIT or CC BY 4.0
(https://synthstrip.io). This file is an independent minimal re-implementation of the
network definition (same layer layout so the published state_dict loads) and pre/post
processing. It does not need FreeSurfer or `surfa`.

Memory: the full-resolution stages are evaluated in overlapping z-slabs so peak RAM stays
around 1-1.5 GB for a whole head (the reference script needs several GB).
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage as ndi

MODEL_DIR = os.environ.get("BODYGLASS_MODELS",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
WEIGHTS = os.path.join(MODEL_DIR, "synthstrip.1.pt")


def available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return os.path.exists(WEIGHTS)


def _build(torch):
    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, act=True):
            super().__init__()
            self.conv = nn.Conv3d(cin, cout, 3, 1, 1)
            self.activation = nn.LeakyReLU(0.2) if act else None

        def forward(self, x):
            x = self.conv(x)
            return self.activation(x) if self.activation is not None else x

    class StripModel(nn.Module):
        # 7-level U-Net, features 16-32-64-64-64-64-64, 2 convs per level, nearest upsampling
        def __init__(self):
            super().__init__()
            feats = np.clip(np.round(16 * 2 ** np.arange(7)).astype(int), 1, 64)
            enc = np.repeat(feats[:-1], 2)
            dec_all = np.repeat(np.flip(feats), 2)
            dec, final = dec_all[:len(enc)], dec_all[len(enc):]
            self.nb_levels = len(enc) // 2 + 1
            prev, enc_nfs = 1, [1]
            self.encoder = nn.ModuleList()
            for lv in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for c in range(2):
                    nf = int(enc[lv * 2 + c]); convs.append(ConvBlock(prev, nf)); prev = nf
                self.encoder.append(convs); enc_nfs.append(prev)
            enc_nfs = enc_nfs[::-1]
            self.decoder = nn.ModuleList()
            for lv in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for c in range(2):
                    nf = int(dec[lv * 2 + c]); convs.append(ConvBlock(prev, nf)); prev = nf
                self.decoder.append(convs)
                prev += enc_nfs[lv]
            self.remaining = nn.ModuleList()
            for nf in final:
                self.remaining.append(ConvBlock(prev, int(nf))); prev = int(nf)
            self.remaining.append(ConvBlock(prev, 1, act=False))

    return StripModel


def _slabbed(fn, x, nconv, slab=32, extra_in=None, up_src=None):
    """Apply fn to x (N,C,D,H,W) in z-slabs with an nconv-voxel halo (exact, not approximate).

    up_src: optional half-resolution tensor that is nearest-upsampled x2 and concatenated
    before x's channels, per slab (avoids materialising it at full resolution)."""
    import torch
    D = x.shape[2]
    out = None
    for z0 in range(0, D, slab):
        z1 = min(D, z0 + slab)
        a, b = max(0, z0 - nconv), min(D, z1 + nconv)
        xs = x[:, :, a:b]
        if up_src is not None:
            ha, hb = a // 2, (b + 1) // 2
            u = torch.nn.functional.interpolate(up_src[:, :, ha:hb], scale_factor=2, mode="nearest")
            u = u[:, :, a - 2 * ha: a - 2 * ha + (b - a)]
            xs = torch.cat([u, xs], 1)
        y = fn(xs)
        if out is None:
            out = torch.empty((x.shape[0], y.shape[1], D, *x.shape[3:]), dtype=y.dtype, device=y.device)
        out[:, :, z0:z1] = y[:, :, z0 - a: z0 - a + (z1 - z0)]
        del y, xs
    return out


def run(grid_ras: np.ndarray, device: str | None = None, border=1.0):
    """grid_ras: 1 mm RAS volume [z,y,x]. Returns (brain_mask bool [z,y,x], info dict).

    Uses Apple-Silicon MPS when available (unsupported ops fall back to CPU), else CPU."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return _run(grid_ras, device, border)
    except Exception:
        if device == "cpu":
            raise
        return _run(grid_ras, "cpu", border)


def _forward(model, x):
    """Same result as the upstream StripModel.forward; the full-resolution stages run in z-slabs."""
    import torch
    enc0 = model.encoder[0]
    h0 = _slabbed(lambda t: enc0[1](enc0[0](t)), x, 2)
    hist = [h0]
    y = torch.nn.functional.max_pool3d(h0, 2)
    for lv in range(1, len(model.encoder)):
        for conv in model.encoder[lv]:
            y = conv(y)
        hist.append(y)
        y = torch.nn.functional.max_pool3d(y, 2)
    nlev = len(model.decoder)
    for lv in range(nlev - 1):
        for conv in model.decoder[lv]:
            y = conv(y)
        y = torch.nn.functional.interpolate(y, scale_factor=2, mode="nearest")
        y = torch.cat([y, hist.pop()], 1)
    for conv in model.decoder[nlev - 1]:
        y = conv(y)
    del h0
    # y is half-res; concat upsampled y with h0 at full res, then the remaining convs, in slabs
    rem = model.remaining

    def tail(t):
        for conv in rem:
            t = conv(t)
        return t
    return _slabbed(tail, hist.pop(), len(rem), up_src=y)


def _run(grid_ras, device, border):
    import time
    import torch
    t0 = time.time()
    dev = torch.device(device)
    model = _build(torch)()
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(dev)

    # --- conform: RAS [z,y,x] -> LIA [i=L, j=I, k=A] (the training orientation), 1 mm
    lia = np.flip(np.flip(np.transpose(grid_ras, (2, 0, 1)), 0), 1)
    # crop to the non-background bounding box (like surfa crop_to_bbox), then centre pad/crop to x64
    # head bounding box (drops empty air / FOV padding so the network input stays small)
    thr = 0.1 * np.percentile(lia, 99) if (lia > 0).any() else 0
    nz = np.nonzero(ndi.binary_opening(lia > thr, iterations=2))
    lo = np.array([n.min() for n in nz]); hi = np.array([n.max() + 1 for n in nz])
    crop = lia[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    target = np.clip(np.ceil(np.array(crop.shape) / 64).astype(int) * 64, 192, 320)
    vol = np.zeros(target, np.float32)
    src_sl, dst_sl = [], []
    for ax, (n, t) in enumerate(zip(crop.shape, target)):
        if n <= t:
            o = (t - n) // 2; src_sl.append(slice(0, n)); dst_sl.append(slice(o, o + n))
        elif ax == 1:  # LIA axis 1 runs head->feet: keep the top of the head, drop the neck
            src_sl.append(slice(0, t)); dst_sl.append(slice(0, t))
        else:
            o = (n - t) // 2; src_sl.append(slice(o, o + t)); dst_sl.append(slice(0, t))
    vol[tuple(dst_sl)] = crop[tuple(src_sl)]
    vol -= vol.min()
    vol = np.clip(vol / max(np.percentile(vol, 99), 1e-6), 0, 1).astype(np.float32)

    # --- forward pass (first and last full-resolution stages in slabs)
    with torch.inference_mode():
        sdt = _forward(model, torch.from_numpy(vol)[None, None].to(dev)).squeeze().float().cpu().numpy()

    # --- unconform back into the RAS grid (SDT outside the crop = far outside the brain)
    full = np.full(lia.shape, 100.0, np.float32)
    region = full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    region[tuple(src_sl)] = sdt[tuple(dst_sl)]
    sdt_ras = np.transpose(np.flip(np.flip(full, 1), 0), (1, 2, 0))
    from .segment import largest_cc
    mask = ndi.binary_fill_holes(largest_cc(sdt_ras < border))
    return mask, {"device": device, "seconds": round(time.time() - t0, 1), "input_shape": [int(s) for s in target]}

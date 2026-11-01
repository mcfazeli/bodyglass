"""Registration QA: (1) T1 edges over each aligned scan (PNG), (2) recovery test: shift/rotate the
moving scan's header by a known amount and check registration brings it back."""
import json, os, sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bodyglass import sequences as S

case = sys.argv[1]; out = sys.argv[2]
mf = json.load(open(os.path.join(case, "manifest.json"))); sj = json.load(open(os.path.join(case, "series.json")))
nx, ny, nz = mf["volume"]["dims"]
t1 = np.fromfile(os.path.join(case, "volume.u8"), np.uint8).reshape(nz, ny, nx).astype(np.float32)
sm = ndi.gaussian_filter(t1, 1.2)
edge = np.hypot(*[ndi.sobel(sm, a) for a in (1, 2)]); edge_c = np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 2))
bc = np.array(mf["brainCenter"]); o = np.array(mf["volume"]["origin"]); c = np.round(bc - o).astype(int)
tiles = []
for q in mf["sequences"]:
    v = np.fromfile(os.path.join(case, q["file"]), np.uint8).reshape(nz, ny, nx)
    row = []
    for sl, e in ((v[c[2]], edge[c[2]]), (v[:, c[1]], edge_c[:, c[1]])):
        # pick slice with data if centre not covered
        g = np.stack([sl] * 3, -1).astype(np.uint8)
        thr = np.percentile(e, 93)
        g[e > thr] = [60, 255, 120]
        row.append(np.flipud(g)[:, ::-1] if sl.shape[0] == ny else np.flipud(g)[:, ::-1])
    h = max(r.shape[0] for r in row); row = [np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0))) for r in row]
    tiles.append((q["kind"], np.concatenate(row, 1)))
W = max(t.shape[1] for _, t in tiles)
img = np.concatenate([np.pad(t, ((0, 4), (0, W - t.shape[1]), (0, 0))) for _, t in tiles], 0)
Image.fromarray(img).save(out)
print("QA image", out, [k for k, _ in tiles])
# recovery test on the directly registered kinds
fixed, fv = S._fixed_image(case, mf); fm = S._fixed_mask(fv); fm.CopyInformation(fixed)
for kind in [k for k in ("flair", "t2", "swi_mag", "dwi") if k in sj]:
    mv = S._moving_image(sj[kind]["files"])
    ref, info0 = S.register(fixed, mv, fm)
    pert = sitk.Euler3DTransform(); pert.SetCenter(ref.GetCenter())
    pert.SetRotation(*np.radians([5, -4, 3])); pert.SetTranslation((6, -5, 4))
    # moved image: resample moving through pert so the true answer is ref∘pert^-1
    moved = sitk.Resample(mv, mv, pert.GetInverse(), sitk.sitkLinear, 0.0)
    tx, info = S.register(fixed, moved, fm, same_frame=True)
    # compare displacement of brain-region points under recovered vs reference
    pts = np.argwhere(sitk.GetArrayFromImage(fm)[::8, ::8, ::8])[:, ::-1] * 8
    P = [fixed.TransformIndexToPhysicalPoint([int(x) for x in p]) for p in pts[::5]]
    err = [np.linalg.norm(np.subtract(tx.TransformPoint(p), pert.TransformPoint(ref.TransformPoint(p)))) for p in P]
    print(f"{kind}: perturbed 5/4/3 deg + 8.8 mm -> residual error mean {np.mean(err):.2f} mm, max {np.max(err):.2f} mm ({info['seconds']} s, rejected={info.get('rejected', False)})")

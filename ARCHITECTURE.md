# BodyGlass architecture

How BodyGlass works inside. For what it does and how to install it, see [README.md](README.md) and
[INSTALL-MAC.md](INSTALL-MAC.md).

BodyGlass is a local macOS app that opens a DICOM export folder, finds the high-resolution 3D T1 series by
metadata, builds 3D surfaces on the Mac and shows them in a browser-based 3D viewer. In the code, "phase 1"
is the quick model (skin, brain surface, white matter, ventricles) and "phase 2" is the detailed
structure labelling that follows in the background.
**Educational only. It makes no diagnostic claims, and a disclaimer is always visible.**

## 1. Stack (and why)
| Part | Choice | Reason |
|---|---|---|
| Processing | Python 3.12: pydicom (+pylibjpeg for compressed DICOM), NumPy/SciPy, scikit-image, trimesh, fast-simplification | Mature, has arm64 wheels, no compiler or GPU needed. FastSurfer is Python/PyTorch too, so it shares the same environment. |
| Brain extraction | **SynthStrip** (Hoopes et al. 2022) weights `synthstrip.1.pt`, 29 MB, MIT or CC BY 4.0; own minimal PyTorch re-implementation in `bodyglass/synthstrip.py` (no FreeSurfer, no `surfa`); MPS on Apple Silicon, CPU otherwise; classic threshold method kept as fallback | Robust across scanners/contrasts; the classic method over-trimmed 2 of 5 test scans. |
| Structure labels (Phase 2) | **FastSurfer v2.6.0** segmentation network only (FastSurferVINN, the `--seg_only` part), **Apache-2.0** code and weights (3 × 22 MB); run as a local subprocess `FastSurferCNN/run_prediction.py` with `--device auto` (MPS on Apple Silicon) and view aggregation on CPU | See §5 for the FastSurfer vs SynthSeg decision. |
| Server | stdlib `http.server`, bound to **127.0.0.1 only** | No framework to install. Not reachable from the network. |
| Viewer | Three.js r186, **vendored** (no CDN). WebGL2, PBR `MeshPhysicalMaterial`, RoomEnvironment IBL, ACES tone mapping, OrbitControls with damping | Gives the "Blender/game" look. Runs in Safari or Chrome on the Mac's GPU. |
| Packaging | `install.sh` (uv + venv) + double-click `BodyGlass.command` | Two steps, no admin rights, no Docker. |

A native wrapper (Tauri/Electron) adds weight and build complexity. A local web page gives the
same rendering, so we skip the wrapper for now.

## 2. Pipeline (runs once per folder; about 40 s on an 8-vCPU Linux machine, CPU only)
```
DICOM folder
 └─ dicom_scan.py  header-only read of every file (no pixels), group by SeriesInstanceUID, score, pick T1
 └─ volume.py      load pixels, sort slices by position along the slice normal, build voxel→LPS affine,
                   resample to an axis-aligned 1 mm RAS grid (handles oblique scans)
 └─ synthstrip.py  conform 1 mm grid to LIA, crop to head, pad to x64, U-Net predicts a signed distance
                   to the brain border → mask = SDT < 1 mm (largest component, holes filled). Full-res
                   stages run in exact overlapping z-slabs, so peak RAM ≈ 1.5–3 GB. ~6–10 s on CPU.
                   Falls back to CPU if an op fails on MPS, and to the classic method if torch/weights missing
                   (or BODYGLASS_STRIP=classic).
 └─ segment.py     with SynthStrip: 3-class Otsu inside the mask (CSF | GM | WM thresholds), brain tissue,
                   white matter, ventricles (dark blobs near the centre that don't reach the brain top).
                   Fallback (classic) estimate: head mask (Otsu + closing + hole fill),
                   WM intensity peak from mid-cranium box → brain = threshold ∩ erode/largest-component/
                   re-dilate, closed brain envelope, white matter (threshold), lateral ventricles
                   (dark CSF deep in envelope), estimated skull shell
 └─ meshes / pipeline.py  marching cubes (brain & WM use the image intensity itself → sub-voxel
                   surfaces), quadric decimation, Taubin smoothing, sulcal-depth vertex shading,
                   export one GLB per layer + volume.u8 (viewer slices) + manifest.json
     ── the quick model is shown now; Phase 2 continues in the background ──
 └─ phase2.py      write the 1 mm grid as NIfTI → FastSurferVINN (3 views × 256 slices, view aggregation)
                   → aparc.DKTatlas+aseg (≈95 labels, conformed LIA 256³) → nearest-neighbour back onto the
                   viewer grid → meshes:
                     · cortex lobes: the Phase 1 brain surface is split by nearest cortical label
                       (31 DKT gyri → frontal/parietal/temporal/occipital, per hemisphere; labels smoothed by
                       an 8-pass majority vote over mesh neighbours), so lobes tile without gaps and keep
                       the sulcal shading; cingulate and insula (not on the outer surface) are closed meshes
                       from their ribbon labels
                     · closed meshes per label group and side: cerebral WM, thalamus, caudate, putamen,
                       pallidum, accumbens, hippocampus, amygdala, ventral DC, lateral ventricles (+ inferior
                       horn, choroid plexus), 3rd/4th ventricle, brainstem, cerebellum (cortex + WM)
                     · parcels.u8: nearest DKT gyrus within 4 mm of cortex, so a click names the gyrus
                   37 GLBs (~0.85–0.95 M triangles, ~16 MB) + manifest.structures / structureInfo / parcels
 └─ server.py      serves web/ and the case folder; JSON API: /api/open, /api/choose-folder (macOS
                   folder dialog via osascript), /api/state (progress), /api/cases
```
Cached cases live in `~/Library/Application Support/BodyGlass/cases/<hash of SeriesInstanceUID>/`
and never leave the Mac. Delete that folder to remove everything.

## 3. Series detection (metadata only, never folder names)
Score per series. Each reason is shown in the UI under "Why this series?".
* **Hard exclusions:** SWI/SWAN (description, sequence name `*swi3d1r`, ImageType `SWI`), phase (`P`),
  mIP/MNIP, diffusion (DWI/TRACEW/ADC/ScanningSequence EP), FLAIR, T2/TSE/PD, localizers, non-MR.
* **Plus points:** MRAcquisitionType=3D (+30), T1/MPRAGE/TFL/BRAVO/SPGR text (+25), GR (+10), IR (+5),
  TI 600–1300 ms (+5), ORIGINAL (+5), slice count (+n/8), near-isotropic ≤1.3 mm (+20), MP2RAGE UNI (+45,
  outweighs its DERIVED flag).
* **Minus points:** DERIVED/SECONDARY/MPR reformat (−40), 2D (−20), slices >3 mm (−15), <60 slices (−25),
  MP2RAGE INV1/INV2 (−50), quantitative T1 map (−30). (`tests/test_series.py` covers these picks.)

On the 11-series test study: MPRAGE 126 points (picked) vs the "T1 SAG 3D_MPR_Axial" reformat at
17 points. Everything else is excluded. A typical clinic sagittal 3D T1 (3D, GR) wins the same way. Every rule is a small, unit-testable function.

## 4. Viewer: quick model
* Smooth orbit, zoom and pan with damping. Turntable mode. Animated view reset.
* **Peel slider** (Skin → Skull → Brain → WM → Deep) plus a visibility toggle and opacity slider per layer.
  Transparent layers use a Fresnel "x-ray glass" shader.
* **Slice planes:** axial, coronal and sagittal planes sample the 3D texture in RAS space, with position
  sliders and contrast. **Cutaway** mode clips every mesh to the camera-facing octant and draws only the
  exposed faces of the slices. There are also three 2D inset views (radiological R/L) with crosshairs;
  click an inset to move the planes.
* **Click-to-pick:** raycast picks a layer (or slice) and shows a card with a plain-language description,
  an "estimated" badge, a rough volume and the hemisphere.
* Disclaimer bar, CSP `default-src 'self'` (the page cannot contact any other host), no telemetry.

### Viewer: labelled structures
* When structures are ready the viewer reloads them in place (the camera stays put). The top bar shows
  "Labelling structures on this computer… nn%" meanwhile.
* Layer rows become groups: Cortex (lobes), White matter, Deep structures, Fluid spaces (ventricles),
  Cerebellum & brainstem. The peel slider fades them from the outside in.
* **Structure list** grouped by layer, one row per structure with a visibility checkbox and L/R chips;
  click a name/chip or the 3D model to select. The card shows a plain-language description
  (`bodyglass/structures.json`), an "automatic label" badge, and volume (grey-matter volume for lobes).
  Clicking a lobe also names the **gyrus under the pointer** (e.g. "Precentral gyrus: the primary motor area…").
* **Isolate** fades everything except the selected structure. **Colour by region** toggles lobe colours vs
  natural brain pink.
* **Floating 3D labels** for the outermost visible group: one per left/right pair (the side nearer the camera).
  Labels hidden behind other structures or a slice are dropped (throttled raycast).
* Cutaway option "Keep deep structures whole in the cut".

## 5. Why FastSurfer for structure labels
| | FastSurfer v2.6.0 (`--seg_only`) | SynthSeg 2.0 |
|---|---|---|
| Licence | Apache-2.0 code **and** weights (Zenodo 10390573) | Apache-2.0 code; robust/parcellation weights distributed via FreeSurfer or a UCL download |
| Apple Silicon | PyTorch MPS (`--device mps`), official macOS arm64 package | TensorFlow/Keras: needs `tensorflow-macos`/`-metal`, a second ML stack next to PyTorch |
| RAM | ~3.5 GB peak measured (CPU, view aggregation on CPU) | several GB with TensorFlow (not measured here) |
| Runtime | 170–280 s CPU here; ~1–2 min expected with MPS | a few minutes on CPU (published figure, not measured here) |
| Output | 95 labels incl. 31 DKT cortical gyri per hemisphere | aseg + optional parcellation |
| Install size | 16 MB code + 67 MB weights, reuses our PyTorch | + TensorFlow (~0.5–1 GB) + weights |

FastSurfer wins on licence clarity of the weights, a single PyTorch stack shared with SynthStrip, and native
MPS support. SynthSeg would be preferable for very low-quality or non-T1 scans (it is contrast-agnostic);
it remains an alternative.
FastSurfer recon-surf could add true pial/white surfaces later (slower, ~30–60 min, needs a FreeSurfer licence
file), which would remove the 1 mm "cauliflower" look.

## 6. Privacy
* Development and testing use **only public sample data** (see "Test data" in README).
* At runtime: the server binds to 127.0.0.1, answers only requests whose Host is `127.0.0.1:<port>` or
  `localhost:<port>` (no DNS rebinding) and refuses POSTs from any other Origin (no cross-site requests from
  other open web pages); the CSP blocks every external origin, and all JS is vendored. Processing and cache stay on
  the local disk. The internet is used only by `install.sh`, by AI model downloads the user starts, and by ChatGPT
  if the user signs in to it (called from the local server, never from the page).
* Structure-labelling weights are downloaded once at install time, never at scan time. (FastSurfer checks for its
  checkpoints at start-up and would only try to download them if they were missing; install.sh places them
  with checksums so this never happens at scan time. FastSurfer runs as a local subprocess on the case folder.)

## 7. Install
See [INSTALL-MAC.md](INSTALL-MAC.md).

## 8. Known limitations
* The benchmarks below are CPU-only; the Apple GPU (MPS) paths are less tested. SynthStrip falls back
  to CPU on error; if FastSurfer fails on MPS, rerun with `BODYGLASS_DEVICE=cpu` (the quick model still works).
* Structure labels are an automatic AI segmentation, unverified per scan. They can be wrong, especially with motion,
  unusual anatomy or lesions. The UI says "automatic label" and gives no normal ranges on purpose.
* Lobe borders on the 3D surface follow mesh triangles, so they look slightly saw-toothed up close.
* Corpus callosum, optic chiasm, hypothalamus (as its own label), and cerebellar sub-lobules are not labelled
  (FastSurfer's CC and CerebNet/HypVINN modules are not run).
* The classic fallback over-trims on some scans (734 and 788 mL on two test sets).
* Quick-model ventricles are approximate (thin/small ventricles may be partly missed or include some cistern CSF);
  the structure labels replace them once ready.
* At 1 mm the brain surface looks "cauliflower"-like, and sulci are only partly resolved.
  The labelled structure surfaces are smoother.
* The skull is an illustration (bone is dark on T1) and is labelled as such.
* The ventricle mask catches the bodies of the lateral ventricles, but not always the full horns.
* The non-AI mask steps (head, skull, depth shading) take ~12–17 s on an 8-vCPU CPU (distance transforms on about 20M voxels). It could be
  cropped or downsampled for speed.
* The non-defaced sample shows a face. Users' own scans will too, which is expected and stays local.
* Compressed DICOM (JPEG/JPEG2000) is supported via pylibjpeg but was not tested with real
  compressed data.
* Enhanced multi-frame MR (one file holding a whole series) is split into its stacks (echoes, magnitude/phase,
  orientations, time points) and each frame is placed from its Per-Frame / Shared Functional Groups. A multi-frame
  file without slice positions, or whose frame count or size doesn't match its pixel data (or would need over
  2 GiB to decode), is listed but not used, with a message saying why. Tested with synthetic files
  only; real scanner exports may hold variations not covered yet.
* Software-WebGL screenshots are slow (about 50 s per frame). On a real GPU the viewer is interactive.

## 9. Benchmarks: brain extraction (8-vCPU Linux, CPU only)
| Dataset (public) | Licence | Scanner | Picked series | SynthStrip mask (mL) | Brain tissue (mL) | Classic method tissue (mL) | SynthStrip time | Total |
|---|---|---|---|---|---|---|---|---|
| open-dicom siemens_mprage_0, in an 11-series test study | none declared | Siemens Prisma 3T | MPRAGE_EnchancedContrast | 1665 | 1232 | 1187 | 9.0 s | 49 s |
| datalad studyforrest T1w | PDDL | Philips 3T | anat-T1w (DERIVED fallback) | 1869 | 1308 | 670 ✗ | 8.0 s | 45 s |
| OpenNeuro ds000001 sub-01 → DICOM | CC0 | Siemens Trio 3T | MPRAGE | 1526 | 1090 | 755 ✗ | 6.6 s | 30 s |
| OpenNeuro ds003097 sub-0001 → DICOM | CC0 | Philips 3T | T1 3D TFE | 1539 | 1246 | 1146 | 5.1 s | 32 s |
| OpenNeuro ds000105 sub-1 → DICOM | PDDL | GE 3T | SAG 3D T1 SPGR | 1391 | 1142 | 1215 | 5.8 s | 24 s |

"Mask" = SynthStrip brain mask including the CSF around the brain; "brain tissue" = grey + white matter
inside it (3-class Otsu). All are rough educational estimates. Reproduce with `tools/run_benchmarks.py`.

## 10. Benchmarks: structure labelling (8-vCPU Linux, CPU only, FastSurfer v2.6.0)
| Dataset (public) | Licence | FastSurfer (s) | Meshing (s) | Total (s) | Seg total (mL) | Cerebral WM | Cerebellum | Thalamus | Hippocampus | Lat. ventricles | Structures / triangles / MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenNeuro ds003097 sub-0001 | CC0 | 174 | 17.6 | 191 | 1215 | 501 | 151 | 17.1 | 9.1 | 13.8 | 37 / 869k / 16.1 |
| OpenNeuro ds000105 sub-1 | PDDL | 182 | 13.1 | 195 | 1110 | 474 | 141 | 15.6 | 9.0 | 20.3 | 37 / 851k / 15.7 |
| OpenNeuro ds000001 sub-01 | CC0 | 170 | 15.7 | 186 | 1153 | 473 | 126 | 15.6 | 8.4 | 7.6 | 37 / 848k / 15.7 |
| datalad studyforrest T1w | PDDL | 184 | 24.7 | 209 | 1410 | 566 | 165 | 16.8 | 9.1 | 26.1 | 37 / 950k / 17.6 |
| siemens_mprage_0 (test_study, 11 series) | none declared | 281 | 26.7 | 307 | 1313 | 517 | 153 | 17.5 | 9.1 | 44.1 | 37 / 913k / 16.9 |

FastSurfer = the 3-view network + view aggregation (child process, peak RSS ~3.5 GB). Meshing = label resampling,
lobe split, 37 GLBs. "Seg total" = all labels except background and extra-cerebral CSF. Volumes are summed left + right.
All numbers are rough educational estimates. Reproduce with `tools/run_phase2_bench.py`.
FastSurfer reports "well under a minute" on an 8 GB GPU and recommends Apple Silicon with 16 GB; Mac timings are
not part of this table.

## 11. Other scan types (FLAIR, T2, TSE, SWI, mIP, DWI, ADC)
* **Detection** (`dicom_scan.classify_kind`, metadata only): ImageType (ADC, TRACEW, DIFFUSION, SWI, MNIP, P),
  DiffusionBValue, ScanningSequence (EP, SE+IR with TI > 1.5 s = FLAIR), SequenceName (`*swi*`, `*tse*`, `spc`)
  and description tokens. Phase images are recognised and skipped. `pick_sequences` keeps one series per
  kind (prefers 3D, coverage, ORIGINAL); `series.json` stores file lists so the alignment can run later.
* **Alignment** (`sequences.py`): fixed = the viewer's T1 grid (RAS -> LPS), head mask; Euler3D, Mattes MI
  (50 bins, 10 % random samples), 3-level pyramid (shrink 4/2/1, smooth 2/1/0 mm), regular-step gradient
  descent. Starts from the scanner coordinates when FrameOfReferenceUID matches (moments otherwise).
  Rejected if it makes the metric worse (both cases; falls back to the starting guess) or, with matching
  scanner coordinates, if it moves > 20 mm or turns > 15° (falls back to header alignment).
  Same-acquisition series share a transform (SWI/mIP <- magnitude, ADC <- DWI).
* **Storage**: `overlays/<kind>.u8` (1–255 robust-scaled; ADC linear 0–3000 ×10⁻⁶ mm²/s; 0 = no data),
  `overlays/sequences.json` (cache, source of truth) mirrored to `manifest.sequences`. Manifest writes from
  structure labelling, alignment and veins go through `manifest_io.update` (lock + re-read + merge of the caller's own
  keys + atomic replace); the quick model writes the first manifest atomically too. A folder that was already
  processed is reused as is (delete the case folder to rebuild).
* **Viewer**: "Scans" panel section appears only when the folder had other scans. Chips T1 / FLAIR / … ;
  "Instead of T1" (grey) or "Colour over T1" (per-scan colour map + strength); uncovered areas are dimmed.
  Overlays are fetched lazily on first use. Compare modal: side by side or swipe, plane tabs, synced
  scrubbing; ADC value on hover. Explainer cards per scan with how it was aligned. Keys S / ⇧S, C.

## 12. Vein tree, measuring, speckle
* **Veins** (`veins.py`): input `overlays/swi.u8` (else mIP) on the viewer grid; ROI = brain zone ∩ SWI coverage eroded 3 mm
  (no-data filled with the median so coverage edges are not 'vessels'); `skimage.filters.frangi(black_ridges=True,
  σ=1,1.5,2, α=β=0.5)`; hysteresis p98.5/p99.7 within ROI; 26-connected parts ≥ 100 voxels; dilate 1, mesh (σ 0.6,
  ≤180k faces). Manifest `veins` {file, source, seconds, components, voxels}. Lower thresholds traced sulcal CSF edges.
* **Measure**: points stored in RAS mm (`centered` frame); 3D spheres + line drawn on top, label at midpoint;
  insets draw points within ±1.5 slices solid, others dashed. Inset pixel ↔ RAS via `insetToRas` / `rasToInset`.
* **Volumes**: info card shows the structure, its other-hemisphere partner and % of the labelled (non-ventricle) brain.
* **Speckle** (`airclean.py`): keep voxels whose 3³ local SD < 0.3·mean+4 and mean > 20 (opened), within 18 mm of
  the brain zone, plus the zone; largest part, closed, holes filled. `manifest.airClean` = version.

## 13. Tour, 3D printing, example brain
* **Modals**: every modal closes with Esc; only one modal is open at a time (`closeAllModals`).
* **Tour**: `tourSteps()` builds 8–12 steps from what is available. Each step sets peel/opacities/selection/planes and
  calls `setView(dir, false, {dur, radius, target})`. URL `?tour=N` is used for screenshots.
* **Export** (`export3d.py`): layers and structures are merged on a 0.6 mm grid, re-meshed, simplified and checked
  watertight; STL or 3MF in millimetres. Whole labelled brain ≈ 7 s, 1 watertight body, 14 MB STL / 1.7 MB 3MF; quick-model brain
  surface ≈ 21 s. Peak memory ≈ 3 GB.
* **Example**: second scene group (same RAS → three transform), materials mirror the user's layer visibility each
  frame; side by side puts the example +x (width + 25 mm); overlay = orange translucent ghost centred on the brain.
  Volume table: per structure (both sides) mL and % of the labelled non-ventricle brain, with a no-diagnosis notice.

## 14. Neck exam (optional, TotalSegmentator)
* **Flow**: ＋ Add scan (or My body → Replace neck) → `server.start_neck` (one job at a time, not while the brain
  pipeline runs) → `neck.build`: pick series (`score_neck`: sagittal, 3D, T1/T2, fine voxels; localizer/DWI/phase/derived
  excluded) → TotalSegmentator `vertebrae_mr` on neck and brain T1 (raw DICOM, not the air-cleaned viewer volume),
  `total_mr --roi_subset spinal_cord intervertebral_discs` on the neck → fit → labels/meshes/volume → `manifest.neck`.
* **Fit**: `fit_rigid` = Kabsch on per-vertebra centroids, then symmetric closest-point (ICP) per vertebra limited to the
  part both fields of view contain. Centroids alone are nearly collinear for C1–C3 (18° roll error on the sample).
  Needs ≥ 2 shared levels; status `ok` if RMS ≤ 3 mm else `refused` (reason text, nothing drawn).
* **Grid**: the brain RAS grid gets `pad` extra slices at the bottom only (shortcut: x/y stay clipped). `neck/volume.u8`
  = neck exam resampled + brain T1 where the brain volume is non-zero (shortcut: visible contrast seam);
  `neck/labels.u8` = merged labels; the viewer pads every other brain-grid volume with `padZ`.
* **Privacy/network**: telemetry off in `models/totalseg/config.json`; the subprocess gets `HTTP(S)_PROXY=127.0.0.1:9`
  (verified: a missing model download fails instead of fetching). Weights only come from `install.sh`.
* **Results (8-vCPU Linux, CPU)**: 45–85 s per neck, peak ≈ 3.7 GB RAM. Same-session sample (truth = identity):
  RMS 0.75 mm, rotation 3.3°, error 1.5 / 3.2 mm (mean / max) at C1–C3, 2.7 / 5.7 mm at C4–T1.
  A different person's neck (LEMON brain + amu01 neck) was accepted at 2.39 mm: the fit cannot detect a wrong patient.


## 15. Adding scans, library, front end
* **Settings** are `BODYGLASS_*` env vars. `install.sh` removes files that older releases leave behind when an update is copied over
  the installed folder. TotalSegmentator is run as `[sys.executable, script]` so stale `#!` lines can't break it.
* **Add scan:** `neck.route(series)` → brain T1 (pick_t1 on non-spine series) and neck (spine-labelled series; all
  series when no brain T1). `POST /api/add` starts the brain build with the neck chained in the same worker, or adds
  a neck to the open model. `POST /api/rebuild` = "Use for 3D instead" (`pipeline.run(uid=…)`, `manifest.sourceFolder`).
* **Library:** a person = a case. `GET /api/cases` returns name/exams/dates; `POST /api/case/rename|delete`,
  `POST /api/reveal` (Finder). All behind the same Host/Origin check.
* **Mouth:** `segment.seal_face` (10 mm closing, only in front of / below the brain centre, ±35 mm of the midline);
  `skinVersion: 2`; `POST /api/reskin` rebuilds only `skin.glb` for models made before the mouth smoothing.
* **Front end:** left rail (Explore/Layers/Scans/About), right inspector on
  selection (Focus / Hide), floating slice bar (same ids), toasts instead of a status chip, welcome screen, step list,
  one automatic tour, ⌘K palette. The inspector is a separate right-hand panel, so the chat panel can sit beside it.

## 16. Ask AI (optional, off by default)
- **Page → local server only.** The chat panel (`#chat`, right column; the inspector shifts left while it is open)
  posts to `/api/ai/*` on 127.0.0.1. Host/Origin checks and the CSP (`connect-src 'self'`) are unchanged: the cloud
  call is made by the Python server, never by the browser.
- **Payload** (`ai_chat.build_payload`): fixed educational instructions + history + one user message built only from
  ticked items; `preview()` renders that same object, `/api/ai/preview` and `/api/ai/chat` share the builder.
- **Backends** (`answer(payload)`): `Local` (MLX `mlx_lm` / `mlx_vlm` in-process, one model in memory at a time,
  llama-cpp-python GGUF fallback; no sockets: tested with a socket/DNS guard) and `Cloud` (`siwc.ChatGPT`: Sign in
  with ChatGPT per developers.openai.com/siwc, scope gate on `chatgpt.tokens.use.direct`, POST /v1/responses with
  `store:false, stream:true`).
- **Models** (`ai_models.Manager`): bundled pinned manifest (`ai_models.json`, made by `tools/pin_ai_models.py`),
  per-file SHA256 (LFS) / git-blob SHA1 checks, Range resume, pause, free-disk check, redirect host allowlist,
  `mlx_lm.convert` / `mlx_vlm.convert` then the originals are deleted; state in `models/state.json`; unfinished
  downloads resume at server start. Data: `~/Library/Application Support/BodyGlass/{models,chatgpt}`.

### Setup, streaming, pictures
- **One-tap setup** (`ai_models.Setup`): `plan()` picks qwen3.5-9b (Apple silicon, ≥ 24 GB), qwen3.5-4b (Apple
  silicon) or the MedPsy GGUF (Intel/Linux, llama.cpp). `start()` checks disk (offers the community 4-bit copy as a
  fix), installs the runtime with the bundled uv and pinned versions (`RUNTIME`; the only PyPI access, user-started),
  then `Manager.start`. `status()` feeds the welcome tile and the % on the Ask AI button; `Manager.eta` = bytes left /
  session speed + a conversion estimate. `install.sh --with-ai-models` is kept as a hidden power-user flag.
- **Streaming**: backends expose `stream(payload, model)` yielding `{"model","note","private"}` then `{"t": text}`.
  `/api/ai/chat` writes NDJSON lines and ends with `{"done": true}` or `{"error","fix"}`. Stop = the page aborts the
  fetch; the next write fails and `gen.close()` stops generation. `ChatError.fix` drives one-tap fixes in the UI
  (setup, image_model, lighter, signin, retry, usage).
- **Pictures**: `Local.pick` answers a message with a picture using an installed vision model when the chosen one is
  text-only (with a note), else `fix: image_model`. ChatGPT gets `input_image`.
- **Page**: chips are built from the screen state (`aiSel`, `meas`, visible layers) plus user items (one picture:
  inset canvas or WebGL `toDataURL`, scaled ≤ 1024 px PNG; report text). Part links: a longest-first regex over
  structure names (and base names without Left/Right) → `focusParts`. Chats: `localStorage bg.chats.<caseId>`.
- **No verdicts**: `INSTRUCTIONS` forbids saying whether the user's own scan/size is normal (only their doctor can
  judge) and asks for 2–5 sentence answers; suggestions never ask "is this typical?". `/api/ai/chat` runs
  `ai_chat.needs_caveat` (phrase regex) on every backend's answer and adds a `{"caveat"}` line shown under it.

# Working on BodyGlass

BodyGlass turns a person's own brain MRI (DICOM) into an interactive 3D model, entirely on their Mac. It is
educational: it never says whether a scan is normal or abnormal. See [README.md](README.md) for what it does and
[ARCHITECTURE.md](ARCHITECTURE.md) for how.

## Stack
- Python 3.12 local server and pipeline (`bodyglass/`): pydicom with pylibjpeg, SimpleITK, SynthStrip,
  FastSurfer, TotalSegmentator, PyTorch; local AI with MLX on Apple Silicon.
- Browser viewer (`bodyglass/web/`): HTML, CSS, JavaScript and three.js (bundled in `vendor/`), WebGL2.
- Setup with `install.sh` (private Python via uv); release zip with `tools/make_dist.sh`.

## Build and test
```bash
./install.sh                                    # on Linux: CPU-only build and model weights
.tools/uv/uv pip install --python .venv/bin/python pytest
./tools/fetch_compressed_samples.sh             # public compressed DICOM samples for the decoder tests
.venv/bin/python -m pytest tests
```
CI (`.github/workflows/tests.yml`) runs the suite on Linux for every pull request and push to `main`, and fails
on skipped tests. Browser tests need Playwright and Chrome and are not run in CI.

## Where things live
- **Spec:** there is no separate spec file. Each issue's acceptance criteria are the spec for that change;
  README and ARCHITECTURE.md describe current behaviour.
- **Style guide:** [CONTRIBUTING.md](CONTRIBUTING.md) (ground rules) and the existing code.
- **Design:** `design/` holds mock-ups and the golden references for design-approved cards.
- **Decisions:** [docs/decisions.md](docs/decisions.md). Add an entry in the same PR when a decision changes.
- **Private notes:** `.context/` in your checkout (gitignored). Nothing private goes in git.

## Privacy rules
- Only public or synthetic data. Never commit anyone's real scans, processed outputs, health data, model weights
  or secrets; `.gitignore` blocks the common types, but check `git status`.
- No faces in screenshots or images.
- No diagnostic claims in the app, docs or AI instructions.
- Keep it local: no new runtime network calls, analytics or CDNs.

## Conventions
- Small changes in the existing structure; avoid new dependencies unless they clearly pay off.
- Keep testable logic in plain Python modules with pytest tests. Add a test for new logic.
- Feature branch and pull request into `main`; never push to `main` directly.

## Workflow and labels
Project board columns: **Todo**, **In Progress**, **In Review**.
- Take the top Todo card that isn't `on-hold` or `needs-design`; move it to In Progress.
- Open a PR into `main` that says `Closes #N`, labelled `ready-for-qa`, and move the card to In Review once CI
  is green.
- Every PR carries proof: a checklist mapping each acceptance criterion to how it's met, a test summary, and
  fixed-size screenshots with synthetic or public data for UI changes.
- QA reviews and merges. A PR needs a judged score of at least 80/100; otherwise it gets `changes-requested`.
- `needs-design` cards wait for a design; `design-approved` cards are built to match the golden reference in
  `design/`. `on-hold` cards are skipped.

# design/

Design references for BodyGlass. Nothing in this folder ships in the app.

| Path | What |
|---|---|
| `STYLE_GUIDE.md` | One-page style guide: colours, type, shape, components, motion and voice, taken from the current UI |
| `ux_mock/` | HTML mockups (1280 × 800) and `render.py`, which renders them next to "before" shots |
| `siwc/` | Reference links for Sign in with ChatGPT |
| `<card-or-feature>/` | Approved design for one card or feature (see below) |

## Golden references
Each approved design lives in `design/<card-or-feature>/` and can hold:

- **Mockup HTML** that links `../ux_mock/base.css` or the style guide's tokens, so it renders without a server.
- **Screenshots** at fixed sizes: 1600 × 960 (the default of `tools/screenshots.py`) and 1280 × 800 (mockups),
  at 1× scale, in Chrome. Put them in a `shots/` folder: `.gitignore` ignores folders named `screenshots/`.
- A short `NOTES.md`: what was approved, by whom, and which issue or card it belongs to.

## Rules
- **Public or synthetic data only.** Use the public sample scans or the example volunteer named in the
  README, or the test study from `tools/make_test_study.py` (public sample plus synthetic series). Never real patient scans, reports, processed
  outputs or chat transcripts.
- **No faces.** Hide the skin or show brain-only views (see CONTRIBUTING.md).
- **Educational, never diagnostic.** Mockup copy follows the same rule as the app: no "normal" or "abnormal",
  no normal ranges.
- Keep mockups local: no CDNs, web fonts or remote images.

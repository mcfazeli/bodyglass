# BodyGlass style guide

One page, taken from the UI as it is now (`bodyglass/web/style.css`, `index.html`, `app.js`,
`structures.json`). **Current** = in the code today, with file refs. **Proposed** = a tidy set to move to;
nothing here changes the app yet.

## Look and feel
A dark, calm "glass" viewer: the 3D model fills the window, and frosted panels float over it. Dark only;
there is no light theme. Keep the model the brightest thing on screen.

## Colour
**Current tokens** (`style.css` `:root`, line 1)

| Token | Value | Role |
|---|---|---|
| `--fg` | `#e9edf5` | Main text |
| `--muted` | `#93a0b6` | Secondary text, section headings |
| `--acc` | `#7cc4ff` | Accent: primary buttons, sliders, "on" states, progress |
| `--glass` | `rgba(18,24,36,.62)` | Panel fill (with `blur(14px) saturate(140%)`) |
| `--line` | `rgba(255,255,255,.08)` | Hairlines and borders |
| — | `#08111e` | Text on accent |
| — | `#1d2636 → #0c1018 → #05070b` | Background radial gradient |

Hard-coded semantic colours: success `#8ff0b0` (chosen series, done steps), warning/disclaimer `#ffe3a8` on
`rgba(40,28,6,.72)`, caution `#ffd58a` (badges, cloud-AI model), measurement `#ffe066`, error `#ff9aa5` /
`#4a1d22` bg, links `#8cc8ff`. Slice planes (`app.js` `PLANE_COLORS`): axial `#5fb6ff`, coronal `#6ee7a0`,
sagittal `#ffb35c`. Anatomy colours live in `structures.json` (e.g. frontal `#e8938f`, hippocampus `#f2d16b`)
and are data, not UI chrome.

**Proposed:** promote the hard-coded values to tokens — `--ok #8ff0b0`, `--warn #ffd58a`, `--warn-fg #ffe3a8`,
`--measure #ffe066`, `--err #ff9aa5`, `--link #8cc8ff`, `--on-acc #08111e`, `--surface rgba(14,19,29,.95)`
(modal fill, currently repeated at .90–.98 opacity in 14 rules), `--accent-soft rgba(124,196,255,.16)`
(selected rows/tabs, currently .12/.14/.16/.2). Merge near-duplicates: `#8cc8ff`/`#8cd3ff` → `--link`;
`#ffd58a`/`#ffd08a`/`#ffe2a8` → `--warn`/`--warn-fg`; `#e9edf5`/`#f2f5fa` → `--fg`.

## Typography
**Current:** system UI stack `-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto`
at **13px / 1.45** (`style.css` line 4); `ui-monospace` for `kbd` and the "exact text sent" box. Twelve sizes
are in use (10–22px: 12px ×15, 10.5/11/11.5/12.5px ×8–9 each) and weights 400/500/600/650/700.
The chat input and report box use `font: … system-ui` instead of the shared stack.

**Proposed scale:** 11 (caption, uppercase section heads at `letter-spacing:1.1px`), 12 (controls, chips),
13 (body), 15 (brand, card titles), 18 (panel/dialog titles), 22 (welcome). Weights: 400 body, 500 secondary
buttons, 600 labels/emphasis, 650 titles. Use `font-variant-numeric: tabular-nums` for every number
(already done for measurements, slice numbers and the example table).

## Shape, spacing, depth
**Current radii:** 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20px. **Proposed:** 6 (kbd, small tags), 9 (buttons,
rows, segmented), 12 (cards, messages, insets), 14 (glass panels), 999 (chips, pills, toasts).
Spacing in practice is a 4px grid (gaps 4–10, panel padding 14–18, screen inset 16, top bar 52px).
Elevation: one shadow, `0 10px 40px rgba(0,0,0,.35)` on `.glass`; stacking via z-index (canvas → labels 3 →
panels 4–5 → disclaimer 6 → dialogs 11–15 → toasts 30 → palette 40).

## Components and patterns
Top bar (brand + ghost buttons + primary split "＋ Add scan"), left icon rail (Explore / Layers / Scans /
About) with a 286px glass panel, right inspector (`#info`, 320px) or Ask AI column (380px), bottom slice bar,
150px slice insets, centred dialogs (help, compare, export, library, models), ⌘K palette, toasts, guided tour,
segmented controls (`.seg`), chips (`.chip`, `.chip.on`), role badges (`.role.ok/.info/.off`), fixed
disclaimer footer. Buttons: **primary** (accent fill, dark text, 600) — one per view; **ghost** (7% white,
hairline) for everything else; `.link` for inline actions.

## Iconography
Unicode glyphs and emoji, no icon font (CSP stays `'self'`): ⌕ ◧ ▤ ⓘ rail, ✦ Ask AI, ▶ Tour, ⟲, ＋, ▾, 📏 ⬇ ⇄ 👤 ✨
in menus. **Proposed:** keep glyphs (local, no assets) but drop emoji in menus for monochrome glyphs so they
take `color` and match the rail.

## Motion
Short and functional: inspector fade/slide `.25s`, progress bars `.3–.4s`, toast fade `.4s`, spinner 1s,
typing caret blink. **Proposed:** add `@media (prefers-reduced-motion: reduce)` to stop the turntable, caret
blink and spinner rotation (none exists today).

## Voice and tone
Plain, calm, second person, short sentences, British-leaning spelling in UI ("Colour by region").
Explain *why* in plain words ("Which scans are used?", then "Technical details" folded away).
**Copy is educational and never diagnostic.** Never say a scan, shape, volume or measurement is normal,
abnormal, healthy, small, large or concerning; no normal ranges. Call results "automatic labels" and
"computer estimates", and point to "Ask your doctor about your results." The disclaimer footer
(`index.html` line 270) and AI instructions follow the same rule. Privacy is stated plainly: "Your scans stay
on this computer."

## Decisions
- **Name:** always **BodyGlass** (capital G), in the app, docs and mockups. "Bodyglass" in `design/ux_mock/`
  is out of date.
- **Accent:** the app's `#7cc4ff` (`--acc`) is canonical. The mockups' `#5aa9ff` is superseded; update the
  mockups when they are next touched.
- **Logo:** the app's all-pink radial dot is canonical (`.logo`, `style.css` line 11): an 18 × 18 px
  circle, `radial-gradient(circle at 35% 35%, #ffd1d6, #d9707f 60%, #7b2c3a)`, glow
  `box-shadow: 0 0 12px #e2788a88`. The mockups' conic-gradient square (`#7cc4ff` → `#b49cff` → `#ff9cc0`,
  `design/ux_mock/base.css` line 7) is superseded.
- **Text colour:** `--fg` stays the app's `#e9edf5` (near-white). The mockups' `#e8edf5` is superseded.

## Open questions
1. `.gitignore` ignores every `screenshots/` folder, so golden screenshots under `design/` need a different
   folder name (proposed `shots/`) or an exception.

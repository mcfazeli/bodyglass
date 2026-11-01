# Contributing

Thanks for your interest in BodyGlass. Issues and pull requests are welcome.

## Ground rules

- **Never commit real patient data.** Use only public sample scans (see "Development" in the README) or
  synthetic data. Don't commit processed outputs, model weights, secrets or build artifacts either;
  `.gitignore` blocks the common file types, but check `git status` before you commit.
- **No faces** in screenshots or images: hide the skin or show brain-only views.
- **No diagnostic claims.** BodyGlass is educational. Text in the app, docs and AI instructions must not say
  whether a scan or measurement is normal or abnormal.
- **Keep it local.** No new runtime network calls, analytics or CDNs. Downloads happen only when the user starts
  them, and the page's Content Security Policy stays `'self'`.
- Keep changes small, follow the existing structure, and avoid new dependencies unless they clearly pay off.

## Before you open a pull request

```bash
.venv/bin/python -m pytest tests
```

All tests must pass. Add a small test for new logic. Describe what changed and why in plain words, and include a
screenshot for visible UI changes.

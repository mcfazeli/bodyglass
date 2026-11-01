#!/bin/bash
# Build /workspace/dist/BodyGlass-mac.zip (code only; weights/data are fetched by install.sh)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"; OUT=/workspace/dist; STAGE="$OUT/stage/BodyGlass"
rm -rf "$OUT/stage" "$OUT/BodyGlass-mac.zip"; mkdir -p "$STAGE/tools"
cd "$SRC"
cp -p install.sh BodyGlass.command *.md LICENSE NOTICE requirements.txt .gitignore "$STAGE/"
mkdir -p "$STAGE/docs" && cp -rp docs/images "$STAGE/docs/"   # README pictures
cp -rp bodyglass "$STAGE/"; find "$STAGE" -name '__pycache__' -prune -exec rm -rf {} +
cp -p tools/*.py tools/make_dist.sh "$STAGE/tools/"
mkdir -p "$STAGE/tests" && cp -p tests/*.py "$STAGE/tests/"
chmod 755 "$STAGE/install.sh" "$STAGE/BodyGlass.command" "$STAGE/tools/make_dist.sh"
python3 - "$OUT/stage" "$OUT/BodyGlass-mac.zip" <<'PY'
import os, sys, zipfile, stat
root, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f); rel = os.path.relpath(p, root)
            zi = zipfile.ZipInfo.from_file(p, rel); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (stat.S_IFREG | (0o755 if os.access(p, os.X_OK) else 0o644)) << 16
            zi.create_system = 3  # unix, so macOS Archive Utility keeps the modes
            with open(p, "rb") as fh: z.writestr(zi, fh.read())
PY
rm -rf "$OUT/stage"
ls -la "$OUT/BodyGlass-mac.zip"

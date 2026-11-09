#!/bin/bash
# BodyGlass one-time setup (macOS on Apple Silicon). Safe to re-run: finished steps are skipped.
# Usage: ./install.sh [--with-ai-models]
# (Ask AI is set up inside the app with one tap. Power users only: --with-ai-models does the same from here.)
#                     You can instead download models later in the app: Ask AI > AI models.
# Needs internet ONCE to fetch packages and model weights. No admin rights, no Homebrew, no system Python:
# everything (uv, Python 3.12, packages, weights) goes inside this folder (.tools, .venv, models, third_party).
# Your scans are never sent anywhere.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
BG_AI=0
for A in "$@"; do case "$A" in --with-ai-models) BG_AI=1;; esac; done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
trap 'die "Setup stopped at line $LINENO. Check your internet connection and run ./install.sh again (it resumes)."' ERR

OS="$(uname -s)"; ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ]; then
  [ "$ARCH" = "arm64" ] || die "This Mac reports '$ARCH'. BodyGlass needs Apple Silicon (M1 or newer). If you are on Apple Silicon, make sure Terminal is not set to 'Open using Rosetta' (Finder > Applications > Utilities > Terminal > Get Info)."
  MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
  [ "$MAJOR" -ge 13 ] || die "macOS 13 (Ventura) or newer is required (PyTorch GPU support). You have $(sw_vers -productVersion)."
  SHA256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "Note: not macOS ($OS/$ARCH). Installing the CPU-only build (developer/test mode)."
  SHA256() { sha256sum "$1" | cut -d' ' -f1; }
fi
command -v curl >/dev/null || die "curl not found."
case "$HERE" in *" "*) echo "Note: the folder path contains spaces; that is fine.";; esac

# --- disk space: ~2.5 GB final (+2 GB with the optional neck labelling), ~4 GB during install
FREE_KB=$(df -Pk "$HERE" | awk 'NR==2{print $4}')
[ "$FREE_KB" -ge 4000000 ] || die "Not enough free disk space: need about 4 GB, have $((FREE_KB/1024/1024)) GB."

# --- updates unzip over this folder without deleting, so remove leftovers of the app's old name
OLDNAME="brain""viewer"   # spelled in two parts so the old name appears nowhere else in the code
rm -rf "./$OLDNAME" "tools/update_from_$OLDNAME.sh" tests/test_migrate.py tests/migrate_install_check.sh

# --- uv (private copy inside this folder; it also downloads its own Python 3.12)
export UV_INSTALL_DIR="$HERE/.tools/uv" UV_PYTHON_INSTALL_DIR="$HERE/.tools/python" UV_CACHE_DIR="$HERE/.tools/cache"
export INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1
UV="$UV_INSTALL_DIR/uv"
if [ ! -x "$UV" ]; then
  say "Installing uv (Python manager, ~40 MB) into .tools/"
  curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh >/dev/null || die "Could not download uv from astral.sh."
fi
[ -x "$UV" ] || die "uv was not installed to $UV."

# --- Python 3.12 environment
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
  echo "Python environment already present."
else
  say "Creating Python 3.12 environment (.venv)"
  rm -rf .venv
  "$UV" venv --python 3.12 --python-preference only-managed .venv || die "Could not create the Python environment."
fi
PY="$HERE/.venv/bin/python"

say "Installing PyTorch (with Apple GPU / MPS support on Mac)"
if [ "$OS" = "Darwin" ]; then
  "$UV" pip install --python "$PY" "torch==2.14.*" "torchvision==0.29.*" || die "PyTorch install failed."
else
  "$UV" pip install --python "$PY" "torch==2.14.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cpu || die "PyTorch install failed."
fi
say "Installing the other packages"
"$UV" pip install --python "$PY" -r requirements.txt || die "Package install failed."

# --- SynthStrip weights (29 MB, MIT / CC BY 4.0, https://synthstrip.io)
fetch() { # url dest sha
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ] && [ "$(SHA256 "$dest")" = "$sha" ]; then echo "OK  $(basename "$dest")"; return; fi
  echo "Downloading $(basename "$dest")..."
  curl -L --fail --retry 3 -# -o "$dest.part" "$url" || die "Download failed: $url"
  [ "$(SHA256 "$dest.part")" = "$sha" ] || { rm -f "$dest.part"; die "Checksum mismatch for $(basename "$dest"). Run ./install.sh again."; }
  mv "$dest.part" "$dest"
}
say "Model weights (checked with SHA-256)"
mkdir -p models
fetch https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt models/synthstrip.1.pt \
  37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33

# --- FastSurfer v2.6.0 (Apache-2.0), segmentation only: code 16 MB + weights 3 x 22 MB
FS=third_party/FastSurfer
# v2.6.0 resolved to this commit. Verify the archive before extracting or executing code.
# GitHub may regenerate archive bytes: a changed digest must fail closed until reviewed.
FS_COMMIT=1dc024735371f65bc4b94a272cf596be6ddf4f21
FS_SHA256=5276fd50636f50d71bf2b5d1d0057b5a8697d5fc290f1a3c81fe7e1b85908613
# Old installs have no marker and must be replaced, even if run_prediction.py exists.
# This marker records verified installation, not protection against later local edits.
if [ ! -f "$FS/FastSurferCNN/run_prediction.py" ] ||
   [ "$(cat "$FS/.bodyglass-archive-sha256" 2>/dev/null || true)" != "$FS_SHA256" ]; then
  (
    echo "Installing verified FastSurfer v2.6.0 code (16 MB)..."
    mkdir -p third_party
    fetch "https://github.com/Deep-MI/FastSurfer/archive/$FS_COMMIT.tar.gz" third_party/fastsurfer.tgz "$FS_SHA256"
    FS_STAGE=$(mktemp -d third_party/.fastsurfer.XXXXXX)
    trap 'rm -rf "$FS_STAGE"' EXIT
    tar xzf third_party/fastsurfer.tgz -C "$FS_STAGE"
    FS_NEW="$FS_STAGE/FastSurfer-$FS_COMMIT"
    [ -f "$FS_NEW/FastSurferCNN/run_prediction.py" ] || die "FastSurfer archive is missing run_prediction.py."
    mkdir -p "$FS_NEW/checkpoints"
    cp -n "$FS"/checkpoints/*.pkl "$FS_NEW/checkpoints/" 2>/dev/null || true
    rm -rf "$FS"
    mv "$FS_NEW" "$FS"
    printf '%s\n' "$FS_SHA256" > "$FS/.bodyglass-archive-sha256"
    rm -f third_party/fastsurfer.tgz
  )
fi
mkdir -p "$FS/checkpoints"
while read -r SHA NAME; do
  fetch "https://zenodo.org/records/10390573/files/$NAME" "$FS/checkpoints/$NAME" "$SHA"
done <<'SUMS'
81ab25ccbfc432cc41fb6089d11ff2a45b5a88e659092096cb1e067269f806b8 aparc_vinn_axial_v2.0.0.pkl
73813957e83ec99d2c4e22f7854308916ec276d66ab7ed8ef8bf3998ba05e289 aparc_vinn_coronal_v2.0.0.pkl
edae6262d69526fea39a019f2dfd8df75eb705f8f025b9b97166e3e17b917414 aparc_vinn_sagittal_v2.0.0.pkl
SUMS

say "Checking the installation"
"$PY" - <<'PYEOF' || die "Self-check failed (see message above)."
import torch, pydicom, trimesh, skimage, nibabel, yacs, SimpleITK, torchvision  # noqa
from bodyglass import synthstrip, phase2
assert synthstrip.available(), "SynthStrip weights missing"
assert phase2.available(), "FastSurfer missing"
mps = torch.backends.mps.is_available()
print(f"PyTorch {torch.__version__}; Apple GPU (MPS): {'yes' if mps else 'no (CPU will be used)'}")
PYEOF
"$UV" cache clean >/dev/null 2>&1 || true
chmod +x BodyGlass.command install.sh
xattr -dr com.apple.quarantine . 2>/dev/null || true
trap - ERR
say "Example brain for comparison (public, CC0; built once on this computer, 2-6 minutes)"
if [ -f example/example.json ]; then echo "OK  example brain already built"
else   # optional step: in a subshell, so a failed download (die) does not fail the whole setup
  mkdir -p example
  if ( fetch https://s3.amazonaws.com/openneuro.org/ds003097/sub-0001/anat/sub-0001_run-1_T1w.nii.gz example/ds003097_sub-0001_T1w.nii.gz \
         e9d16f1c58d38ce1695873ad855a05299a0502184d98187ce28ef8a433d12ec0 ) &&
     "$PY" tools/build_example.py example/ds003097_sub-0001_T1w.nii.gz > example/build.log 2>&1; then
    rm -f example/ds003097_sub-0001_T1w.nii.gz; echo "OK  example brain built"
  else echo "Note: the example brain could not be added (see the message above or example/build.log). Everything else works; run ./install.sh again to retry."; fi
fi

say "Neck labelling (optional; TotalSegmentator 2.18, Apache-2.0: ~1.3 GB packages + ~0.7 GB model weights)"
TS_HOME="$HERE/models/totalseg"
if [ -f "$TS_HOME/ready" ]; then echo "OK  neck labelling already installed"
# optional step in a subshell like the example: a failed download only leaves the neck option "unavailable"
# (no "set -e" here: bash ignores it inside an if-condition, so every step is chained with &&)
elif ( mkdir -p "$TS_HOME" &&
       # telemetry off before TotalSegmentator ever runs (it would create a config with usage stats ON)
       printf '{"totalseg_id": "totalseg_local", "send_usage_stats": false, "prediction_counter": 0}\n' > "$TS_HOME/config.json" &&
       "$UV" pip install --python "$PY" "TotalSegmentator==2.18.0" &&
       TOTALSEG_HOME_DIR="$TS_HOME" .venv/bin/totalseg_download_weights -t total_mr &&
       TOTALSEG_HOME_DIR="$TS_HOME" .venv/bin/totalseg_download_weights -t vertebrae_mr &&
       # 852 = total_mr's 3 mm model, used to crop to the cord/discs (--roi_subset); not in the total_mr set
       TOTALSEG_HOME_DIR="$TS_HOME" "$PY" -c "from totalsegmentator.libs import download_pretrained_weights as d; d(852)" &&
       # only mark ready when all four trained models are really on disk
       # shortcut: a half-unzipped model folder is not deleted, so a retry may need models/totalseg removed by hand
       for D in 756 850 851 852; do ls "$TS_HOME"/nnunet/results/Dataset${D}_*/*/fold_0/checkpoint_final.pth >/dev/null || exit 1; done &&
       "$PY" -c "import totalsegmentator" && touch "$TS_HOME/ready" ); then
  "$UV" cache clean >/dev/null 2>&1 || true; echo "OK  neck labelling installed"
else echo "Note: neck labelling could not be added (see the message above). Everything else works; adding a neck scan shows as unavailable. Run ./install.sh again to retry."; fi

if [ "$BG_AI" = 1 ]; then
  say "Private AI (power-user flag; the app's Ask AI > Set up does the same): engine, then the default model"
  if [ "$(uname -m)" = arm64 ]; then AI_PKGS="mlx-lm==0.32.2 mlx-vlm==0.7.8"; else AI_PKGS="llama-cpp-python==0.3.36"; fi
  # shellcheck disable=SC2086
  if "$UV" pip install --python "$PY" $AI_PKGS; then
    if [ "$(uname -m)" = arm64 ]; then "$PY" -m bodyglass.ai_models qwen3.5-4b 4 || echo "Note: the model download stopped; finish it in the app (Ask AI > Set up)."
    else "$PY" -m bodyglass.ai_models medpsy-4b-gguf 4 || echo "Note: the model download stopped; finish it in the app (Ask AI > Set up)."; fi
  else echo "Note: the AI runtime could not be installed. Everything else works; use Ask AI > Set up in the app."; fi
fi

say "Done. Double-click BodyGlass.command (or run ./BodyGlass.command) to start."

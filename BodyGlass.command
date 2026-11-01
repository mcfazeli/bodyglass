#!/bin/bash
# Double-click launcher (macOS). Starts BodyGlass and opens it in your browser.
# Everything runs on this Mac at http://127.0.0.1:8765 - nothing is uploaded. Close this window to quit.
cd "$(dirname "$0")"
if ! .venv/bin/python -c '' 2>/dev/null || [ ! -f models/synthstrip.1.pt ]; then ./install.sh || { echo "Setup failed. Press Enter to close."; read -r; exit 1; }; fi
exec .venv/bin/python -m bodyglass "$@"

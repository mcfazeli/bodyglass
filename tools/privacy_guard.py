"""Fail if a tracked file looks like a scan, a mesh, model weights or a secret.

Path rules come from the Privacy, Model weights and Secrets sections of .gitignore, so `git add -f` can't sneak
one past. Content rules catch DICOM, gzip and NIfTI files under any name. A file listed in
tests/public_samples.txt (path, source URL, licence) is allowed.

Usage: python tools/privacy_guard.py   (run from the repo root; exits 1 and lists every problem)
"""
import fnmatch
import subprocess
import sys
from pathlib import Path

SECTIONS = ("Privacy", "Model weights", "Secrets")
ALLOWLIST = "tests/public_samples.txt"


def guarded_patterns(gitignore: str) -> list[str]:
    """Patterns under a `# --- <section>` header named in SECTIONS, up to the next comment line."""
    patterns, keep = [], False
    for line in gitignore.splitlines():
        line = line.strip()
        if line.startswith("#"):
            keep = line.startswith("# ---") and any(s in line for s in SECTIONS)
        elif line and keep:
            patterns.append(line)
    return patterns


def path_rule(path: str, patterns: list[str]) -> str | None:
    parts = path.split("/")
    for p in patterns:
        # shortcut: no gitignore pattern here has an inner slash or `!`; extend this if one is added.
        names = parts[:-1] if p.endswith("/") else parts
        if any(fnmatch.fnmatchcase(n, p.rstrip("/")) for n in names):
            return f"matches .gitignore pattern '{p}'"
    return None


def content_rule(head: bytes) -> str | None:
    if head[128:132] == b"DICM":
        return "is a DICOM file (DICM at byte 128)"
    if head[:2] == b"\x1f\x8b":
        return "is gzip-compressed (could be a .nii.gz scan)"
    size = int.from_bytes(head[:4], "little"), int.from_bytes(head[:4], "big")
    if 348 in size and head[344:348] in (b"n+1\0", b"ni1\0"):
        return "is a NIfTI-1 image"
    if 540 in size and head[4:8] in (b"n+2\0", b"ni2\0"):
        return "is a NIfTI-2 image"
    return None


def read_allowlist(text: str) -> tuple[set[str], list[str]]:
    """Entries are `path  https://source  licence`; returns the allowed paths and any malformed lines."""
    allowed, errors = set(), []
    for n, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 3 or not fields[1].startswith(("https://", "http://")):
            errors.append(f"{ALLOWLIST}:{n}: needs 'path source-URL licence' on one line: {line.strip()}")
        else:
            allowed.add(fields[0])
    return allowed, errors


def check(root: Path, paths: list[str]) -> list[str]:
    gitignore = root / ".gitignore"
    allow_file = root / ALLOWLIST
    patterns = guarded_patterns(gitignore.read_text() if gitignore.exists() else "")
    allowed, errors = read_allowlist(allow_file.read_text() if allow_file.exists() else "")
    for path in paths:
        if path in allowed:
            continue
        rule = path_rule(path, patterns)
        file = root / path
        if rule is None and file.is_file() and not file.is_symlink():
            with open(file, "rb") as f:
                rule = content_rule(f.read(348))
        if rule:
            errors.append(f"{path}: {rule}")
    return errors


def main() -> int:
    root = Path.cwd()
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    errors = check(root, [p for p in out.decode().split("\0") if p])
    for e in errors:
        print(f"::error::{e}")
    if errors:
        print(f"Privacy guard: {len(errors)} problem(s). Remove the file, or if it is a public sample, add it to "
              f"{ALLOWLIST} with its source URL and licence (see CONTRIBUTING.md).")
        return 1
    print("Privacy guard: no scans, meshes, weights or secrets tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

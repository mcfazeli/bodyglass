"""The CI privacy guard flags scans, secrets and bad allowlist lines, and lets listed public samples through."""
import gzip
from pathlib import Path

from tools import privacy_guard as guard

GITIGNORE = """__pycache__/
# --- Privacy: never commit scans or processed outputs ---
*.nii.gz
DICOM/
# --- Model weights ---
*.pt
# --- Secrets ---
*token*.json
# --- Build ---
*.zip
"""


def make_repo(tmp_path: Path, files: dict[str, bytes], allowlist: str = "") -> list[str]:
    (tmp_path / ".gitignore").write_text(GITIGNORE)
    (tmp_path / "tests").mkdir()
    (tmp_path / guard.ALLOWLIST).write_text(allowlist)
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
    return guard.check(tmp_path, list(files))


DICOM = b"\0" * 128 + b"DICM" + b"\x02\x00" * 20


def test_reads_only_the_guarded_gitignore_sections():
    assert guard.guarded_patterns(GITIGNORE) == ["*.nii.gz", "DICOM/", "*.pt", "*token*.json"]


def test_renamed_dicom_is_caught_by_content(tmp_path):
    errors = make_repo(tmp_path, {"notes/scan.txt": DICOM, "README.md": b"# hi\n"})
    assert errors == ["notes/scan.txt: is a DICOM file (DICM at byte 128)"]


def test_nifti_token_weights_and_dicom_folder_are_caught_by_path(tmp_path):
    errors = make_repo(tmp_path, {
        "brain.nii.gz": gzip.compress(b"x"),
        "auth/token.json": b"{}",
        "model.pt": b"w",
        "DICOM/IM0001": b"x",
        "build.zip": b"PK",  # build section, not guarded
    })
    assert sorted(errors) == [
        "DICOM/IM0001: matches .gitignore pattern 'DICOM/'",
        "auth/token.json: matches .gitignore pattern '*token*.json'",
        "brain.nii.gz: matches .gitignore pattern '*.nii.gz'",
        "model.pt: matches .gitignore pattern '*.pt'",
    ]


def test_gzip_and_nifti_headers_under_other_names(tmp_path):
    nifti1 = (348).to_bytes(4, "little") + b"\0" * 340 + b"n+1\0"
    errors = make_repo(tmp_path, {"a.dat": gzip.compress(b"x"), "b.dat": nifti1})
    assert sorted(errors) == ["a.dat: is gzip-compressed (could be a .nii.gz scan)", "b.dat: is a NIfTI-1 image"]


def test_allowlisted_public_sample_passes(tmp_path):
    allow = "# comment\ntests/data/sample.dcm https://example.org/sample.dcm CC0-1.0\n"
    assert make_repo(tmp_path, {"tests/data/sample.dcm": DICOM}, allow) == []


def test_allowlist_line_without_licence_fails(tmp_path):
    allow = "tests/data/sample.dcm https://example.org/sample.dcm\n"
    errors = make_repo(tmp_path, {"tests/data/sample.dcm": DICOM}, allow)
    assert errors[0].startswith(f"{guard.ALLOWLIST}:1: needs 'path source-URL licence'")
    assert errors[1] == "tests/data/sample.dcm: is a DICOM file (DICM at byte 128)"  # not allowed, so still checked


"""Exercise the real installer blocks with synthetic archives and no network/model installs."""
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
COMMIT = re.search(r"^FS_COMMIT=([0-9a-f]{40})$", SOURCE, re.M)[1]
PIN = re.search(r"^FS_SHA256=([0-9a-f]{64})$", SOURCE, re.M)[1]
FETCH = SOURCE[SOURCE.index("fetch() {"):SOURCE.index('say "Model weights')]
BLOCK = SOURCE[SOURCE.index("FS=third_party/FastSurfer"):SOURCE.index('mkdir -p "$FS/checkpoints"\nwhile')]


def archive(path, *, valid=True):
    with tarfile.open(path, "w:gz") as tar:
        data = b"# synthetic verified code\n"
        info = tarfile.TarInfo(f"FastSurfer-{COMMIT}/FastSurferCNN/" +
                                ("run_prediction.py" if valid else "other.py"))
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tmp_path, payload, digest):
    # Stub only transport, retaining production hash verification and real tar extraction.
    script = '''set -euo pipefail
    die() { echo "$*" >&2; exit 1; }
    SHA256() { shasum -a 256 "$1" | cut -d' ' -f1; }
    curl() {
      printf '%s\\n' "$*" >> downloads.log
      while [ "$1" != -o ]; do shift; done
      cp "$PAYLOAD" "$2"
    }
    ''' + FETCH + BLOCK.replace(PIN, digest)
    return subprocess.run(["bash", "-c", script], cwd=tmp_path,
                          env={**os.environ, "PAYLOAD": str(payload)},
                          capture_output=True, text=True)


def installed(tmp_path):
    fs = tmp_path / "third_party/FastSurfer"
    (fs / "FastSurferCNN").mkdir(parents=True)
    (fs / "FastSurferCNN/run_prediction.py").write_text("old unverified code")
    (fs / "checkpoints").mkdir()
    (fs / "checkpoints/model.pkl").write_text("preserve weights")
    return fs


@pytest.mark.parametrize("existing", [False, True])
def test_verified_install_and_rerun(tmp_path, existing):
    payload = tmp_path / "fixture.tgz"
    digest = archive(payload)
    fs = installed(tmp_path) if existing else tmp_path / "third_party/FastSurfer"
    result = run(tmp_path, payload, digest)
    assert result.returncode == 0, result.stderr
    assert (fs / "FastSurferCNN/run_prediction.py").read_text() == "# synthetic verified code\n"
    assert (fs / ".bodyglass-archive-sha256").read_text().strip() == digest
    if existing:
        assert (fs / "checkpoints/model.pkl").read_text() == "preserve weights"
    assert not list((tmp_path / "third_party").glob(".fastsurfer.*"))
    assert not (tmp_path / "third_party/fastsurfer.tgz").exists()
    assert run(tmp_path, payload, digest).returncode == 0
    downloads = (tmp_path / "downloads.log").read_text().splitlines()
    assert len(downloads) == 1
    assert f"/archive/{COMMIT}.tar.gz" in downloads[0]


def test_bad_download_never_extracted_and_old_install_preserved(tmp_path):
    payload = tmp_path / "fixture.tgz"
    archive(payload)
    fs = installed(tmp_path)
    result = run(tmp_path, payload, "0" * 64)
    assert result.returncode != 0
    assert "Checksum mismatch" in result.stderr
    assert (fs / "FastSurferCNN/run_prediction.py").read_text() == "old unverified code"
    assert not (fs / ".bodyglass-archive-sha256").exists()
    assert not (tmp_path / "third_party/fastsurfer.tgz.part").exists()
    assert not list((tmp_path / "third_party").glob(".fastsurfer.*"))


@pytest.mark.parametrize("cached_valid", [False, True])
def test_cache_verified_before_use(tmp_path, cached_valid):
    payload = tmp_path / "fixture.tgz"
    digest = archive(payload)
    (tmp_path / "third_party").mkdir()
    cache = tmp_path / "third_party/fastsurfer.tgz"
    cache.write_bytes(payload.read_bytes() if cached_valid else b"corrupted archive")
    result = run(tmp_path, payload, digest)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "downloads.log").exists() != cached_valid


@pytest.mark.parametrize("malformed", [False, True])
def test_invalid_archive_preserves_existing_install(tmp_path, malformed):
    payload = tmp_path / "fixture.tgz"
    digest = archive(payload, valid=False)
    if malformed:
        payload.write_bytes(b"not a tarball")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    fs = installed(tmp_path)
    assert run(tmp_path, payload, digest).returncode != 0
    assert (fs / "FastSurferCNN/run_prediction.py").read_text() == "old unverified code"
    assert not (fs / ".bodyglass-archive-sha256").exists()
    assert not list((tmp_path / "third_party").glob(".fastsurfer.*"))


def test_stale_marker_or_missing_code_triggers_repair(tmp_path):
    payload = tmp_path / "fixture.tgz"
    digest = archive(payload)
    fs = installed(tmp_path)
    (fs / ".bodyglass-archive-sha256").write_text("old digest\n")
    assert run(tmp_path, payload, digest).returncode == 0
    (fs / "FastSurferCNN/run_prediction.py").unlink()
    assert run(tmp_path, payload, digest).returncode == 0
    assert (fs / "FastSurferCNN/run_prediction.py").exists()
    assert len((tmp_path / "downloads.log").read_text().splitlines()) == 2

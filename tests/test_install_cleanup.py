"""install.sh removes the old package folder an update unzips over (rsync without --delete keeps it)."""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_install_removes_stale_old_package(tmp_path):
    src = open(os.path.join(ROOT, "install.sh")).read()
    block = src[src.index("# --- updates unzip"):src.index("# --- uv (private copy")]
    old = "brain" + "viewer"
    for d in (old, "bodyglass", "tools"):
        os.makedirs(tmp_path / d)
    (tmp_path / old / "__init__.py").write_text("")
    (tmp_path / "bodyglass" / "__init__.py").write_text("")
    (tmp_path / "tools" / f"update_from_{old}.sh").write_text("")
    subprocess.run(["bash", "-euc", block], cwd=tmp_path, check=True)
    assert not (tmp_path / old).exists() and not (tmp_path / "tools" / f"update_from_{old}.sh").exists()
    assert (tmp_path / "bodyglass" / "__init__.py").exists()

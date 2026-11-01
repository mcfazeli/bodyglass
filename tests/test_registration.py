"""Rigid registration recovers a known header offset, and the sanity limits reject implausible fits."""
import os

import numpy as np
import pytest
from scipy import ndimage as ndi

sitk = pytest.importorskip("SimpleITK")
from bodyglass.sequences import register  # noqa: E402


def phantom():
    rng = np.random.default_rng(1)
    z, y, x = np.mgrid[:64, :64, :64] - 32.0
    head = ((x / 26) ** 2 + (y / 22) ** 2 + (z / 24) ** 2 < 1).astype(np.float32)
    inner = ndi.gaussian_filter(rng.random((64, 64, 64)).astype(np.float32), 3)
    return head * (50 + 200 * (inner - inner.min()) / np.ptp(inner))


def img(a, origin):
    im = sitk.GetImageFromArray(a)
    im.SetSpacing((2.0, 2.0, 2.0))
    im.SetOrigin(origin)
    return im


def test_recovers_known_shift_same_frame():
    a = phantom()
    fixed, moving = img(a, (0, 0, 0)), img(a, (4.0, -3.0, 2.0))   # same pixels, header moved by d
    tx, info = register(fixed, moving, same_frame=True)
    assert not info.get("rejected")
    assert np.allclose(tx.GetTranslation(), (4.0, -3.0, 2.0), atol=0.6), tx.GetTranslation()


def test_large_jump_rejected_for_same_frame():
    a = phantom()
    tx, info = register(img(a, (0, 0, 0)), img(a, (30.0, 0, 0)), same_frame=True)
    assert info["rejected"] and np.allclose(tx.GetTranslation(), 0)   # falls back to scanner coordinates


def test_different_frame_large_offset_found_from_centre_of_mass():
    a = phantom()
    tx, info = register(img(a, (0, 0, 0)), img(a, (30.0, -20.0, 10.0)), same_frame=False)
    assert not info.get("rejected")
    assert np.allclose(tx.GetTranslation(), (30.0, -20.0, 10.0), atol=0.6)


NECK = os.path.join(os.path.dirname(__file__), "..", "sample_data", "neck_ds001919", "")


@pytest.mark.skipif(not os.path.exists(NECK + "sub-amu01_T2w.nii.gz"), reason="public neck sample not downloaded")
def test_thin_slab_that_slides_off_is_rejected_even_when_metric_improves():
    """ds001919 sub-amu01 (CC0): a 51 mm sagittal T2 slab starts exactly aligned to the head+neck T1, yet
    brain-tuned rigid MI 'improves' the metric while drifting tens of mm. It must be rejected."""
    fixed = sitk.ReadImage(NECK + "sub-amu01_T1w.nii.gz", sitk.sitkFloat32)
    moving = sitk.ReadImage(NECK + "sub-amu01_T2w.nii.gz", sitk.sitkFloat32)
    tx, info = register(fixed, moving, same_frame=False)
    assert info["rejected"], info

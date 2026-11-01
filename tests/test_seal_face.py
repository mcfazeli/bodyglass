"""Mouth closing: a hole in the lower face is bridged; the back of the head and the brain stay exactly as they were."""
import numpy as np
from scipy import ndimage as ndi

from bodyglass.segment import seal_face


def test_mouth_notch_closed_back_and_brain_untouched():
    z, y, x = np.ogrid[:120, :140, :120]
    head = ((z - 60) / 55.0) ** 2 + ((y - 70) / 65.0) ** 2 + ((x - 60) / 50.0) ** 2 <= 1
    brain = ((z - 75) / 30.0) ** 2 + ((y - 65) / 38.0) ** 2 + ((x - 60) / 30.0) ** 2 <= 1
    mouth = (abs(z - 35) < 6) & (y > 112) & (abs(x - 60) < 12)        # open mouth: front, below the brain
    back = (abs(z - 35) < 6) & (y < 25) & (abs(x - 60) < 12)          # same notch at the back of the head
    cut = head & ~mouth & ~back
    out = seal_face(cut, (75, 65, 60))
    deep = mouth & (ndi.distance_transform_edt(head) > 4)
    assert out[deep].all() and out[mouth & head].mean() > 0.8           # sealed; only a ~1-2 mm dip at the lips remains
    assert not out[back & head].any()                                  # back notch left alone
    assert (out[brain] == cut[brain]).all() and (out | ~cut).all()     # brain region unchanged, nothing removed
    assert (out & ~head).sum() < 0.01 * head.sum()                     # no bulge beyond the original outline

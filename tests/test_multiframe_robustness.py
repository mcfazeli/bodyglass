"""QA: malformed enhanced multi-frame files. All DICOM here is synthetic, built in the test.
Tests marked xfail(strict) document known gaps found in review of #15; they flip to failures once fixed,
so remove the marker then."""
import numpy as np
import pydicom
import pytest
from pydicom.uid import RLELossless

from bodyglass.dicom_scan import pick_t1, scan_folder
from bodyglass.volume import load_series
from test_multiframe import COL_DIR, N, P0, PS, ROW_DIR, TRUTH, write_enhanced


def _edit(path, fn):
    ds = pydicom.dcmread(path)
    fn(ds)
    ds.save_as(path, enforce_file_format=False)


def test_truncated_header_is_not_used(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    p.write_bytes(p.read_bytes()[:1500])
    found = scan_folder(str(tmp_path))   # cut inside the per-frame groups: a few frames may be listed, never picked
    assert pick_t1(found) is None and all(len(s.files) < N for s in found)


def test_missing_functional_groups_are_rejected_not_crashed(tmp_path):
    for name, seq in (("a", "PerFrameFunctionalGroupsSequence"), ("b", "SharedFunctionalGroupsSequence")):
        (tmp_path / name).mkdir()
        write_enhanced(tmp_path / name / "IM0001")
        _edit(tmp_path / name / "IM0001", lambda d: delattr(d, seq))
        (s,) = scan_folder(str(tmp_path / name))
        assert s.excluded == "multi-frame file without slice positions"


def test_rle_compressed_multiframe_loads(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: d.compress(RLELossless, d.pixel_array))
    (s,) = scan_folder(str(tmp_path))
    vol, aff = load_series(s.files)
    assert np.array_equal(vol, TRUTH * 2.0 - 10) and np.allclose(aff[:3, 3], P0)


@pytest.mark.xfail(strict=True, reason="NumberOfFrames is trusted: a header-only file can claim millions of frames "
                                       "and scan_folder loops over all of them")
def test_frame_count_beyond_per_frame_groups_is_not_trusted(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "NumberOfFrames", 20_000))   # pixel data and groups still hold N frames
    found = scan_folder(str(tmp_path))
    assert all(len(s.files) <= N for s in found)


@pytest.mark.xfail(strict=True, reason="top-level ImagePositionPatient overrides per-frame positions, so all frames "
                                       "collapse to one slice after the scan reported 64 images")
def test_stale_top_level_position_does_not_collapse_the_volume(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "ImagePositionPatient", list(P0)))
    (s,) = scan_folder(str(tmp_path))
    vol, _ = load_series(s.files)
    assert vol.shape[0] == N


@pytest.mark.xfail(strict=True, reason="a multi-frame file with only top-level geometry is picked as the T1 but "
                                       "loads as a single slice")
def test_multiframe_with_one_shared_position_is_not_picked(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)

    def legacy(d):
        del d.PerFrameFunctionalGroupsSequence, d.SharedFunctionalGroupsSequence
        d.ImagePositionPatient, d.ImageOrientationPatient = list(P0), [*ROW_DIR, *COL_DIR]
        d.PixelSpacing, d.SliceThickness, d.MRAcquisitionType = PS, 1.2, "3D"
    _edit(p, legacy)
    assert pick_t1(scan_folder(str(tmp_path))) is None


@pytest.mark.xfail(strict=True, reason="NumberOfFrames larger than the pixel data is only caught at load time")
def test_frame_count_larger_than_pixel_data_is_caught_at_scan(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "NumberOfFrames", 200))
    found = scan_folder(str(tmp_path))
    assert all(s.excluded for s in found)

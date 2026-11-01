"""Malformed enhanced multi-frame files. All DICOM here is synthetic, built in the test."""
import numpy as np
import pydicom
import pytest
from pydicom.uid import RLELossless

from bodyglass.dicom_scan import BAD_FRAMES, pick_t1, scan_folder
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


def test_frame_count_beyond_per_frame_groups_is_not_trusted(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "NumberOfFrames", 20_000))   # pixel data and groups still hold N frames
    found = scan_folder(str(tmp_path))
    assert all(len(s.files) <= N for s in found)


def test_stale_top_level_position_does_not_collapse_the_volume(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "ImagePositionPatient", list(P0)))
    (s,) = scan_folder(str(tmp_path))
    vol, _ = load_series(s.files)
    assert vol.shape[0] == N


def test_multiframe_with_one_shared_position_is_not_picked(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)

    def legacy(d):
        del d.PerFrameFunctionalGroupsSequence, d.SharedFunctionalGroupsSequence
        d.ImagePositionPatient, d.ImageOrientationPatient = list(P0), [*ROW_DIR, *COL_DIR]
        d.PixelSpacing, d.SliceThickness, d.MRAcquisitionType = PS, 1.2, "3D"
    _edit(p, legacy)
    assert pick_t1(scan_folder(str(tmp_path))) is None


def test_frame_count_larger_than_pixel_data_is_caught_at_scan(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: setattr(d, "NumberOfFrames", 200))
    found = scan_folder(str(tmp_path))
    assert all(s.excluded for s in found)


def test_huge_claimed_image_size_is_refused_before_decoding(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: (d.compress(RLELossless, d.pixel_array), setattr(d, "Rows", 60_000), setattr(d, "Columns", 60_000)))
    (s,) = scan_folder(str(tmp_path))
    assert s.excluded == "damaged or oversized multi-frame file"
    with pytest.raises(RuntimeError, match="too large to open safely"):
        load_series([f"{p}#frame=0"])
    assert "too large to open safely" in BAD_FRAMES


def test_frames_sharing_one_position_are_not_picked(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: [g.PlanePositionSequence[0].__setattr__("ImagePositionPatient", list(P0))
                        for g in d.PerFrameFunctionalGroupsSequence])
    (s,) = scan_folder(str(tmp_path))
    assert s.excluded == "multi-frame file without slice positions"


def test_classic_file_named_like_a_frame_reference_still_loads(tmp_path):
    from test_multiframe import write_classic
    write_classic(tmp_path)
    for f in tmp_path.iterdir():
        f.rename(tmp_path / (f.name + "#frame=3"))
    (s,) = scan_folder(str(tmp_path))
    assert load_series(s.files)[0].shape == (N, 6, 5)


def test_classic_files_with_number_of_frames_one_are_still_used(tmp_path):
    from test_multiframe import write_classic
    write_classic(tmp_path)
    for f in tmp_path.glob("im*.dcm"):
        _edit(f, lambda d: setattr(d, "NumberOfFrames", 1))
    (s,) = scan_folder(str(tmp_path))
    assert s.excluded is None and pick_t1([s]) is s


def test_truncated_pixel_data_is_caught_at_scan(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    p.write_bytes(p.read_bytes()[:-100])   # header intact, last frames cut off
    (s,) = scan_folder(str(tmp_path))
    assert s.excluded == "damaged or oversized multi-frame file"
    with pytest.raises(RuntimeError, match="looks damaged"):
        load_series([f"{p}#frame=0"])


def test_compressed_frames_that_do_not_decode_get_the_plain_message(tmp_path):
    p = tmp_path / "IM0001"
    write_enhanced(p)
    _edit(p, lambda d: (d.compress(RLELossless, d.pixel_array), setattr(d, "Rows", 600)))   # still under the cap
    (s,) = scan_folder(str(tmp_path))
    with pytest.raises(RuntimeError, match="looks damaged"):
        load_series(s.files)

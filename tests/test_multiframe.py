"""Enhanced multi-frame MR (one file, many frames) loads with the same geometry as a classic one-file-per-slice
series. All DICOM here is synthetic, built in the test."""
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from bodyglass import server
from bodyglass.dicom_scan import pick_t1, scan_folder
from bodyglass.volume import load_series

N, ROWS, COLS = 64, 6, 5
PS, STEP = [0.9, 0.8], 1.2                    # row spacing, column spacing, slice step (mm)
a = np.deg2rad(20)                            # oblique sagittal: tilted 20 degrees
ROW_DIR, COL_DIR = np.array([0, np.cos(a), np.sin(a)]), np.array([0, -np.sin(a), np.cos(a)])
NORMAL, P0 = np.cross(ROW_DIR, COL_DIR), np.array([-40.0, -90.0, -60.0])
TRUTH = np.random.default_rng(0).integers(0, 4000, (N, ROWS, COLS)).astype(np.uint16)


def _base(sop_class, series_uid):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID, ds.Modality, ds.SeriesDescription = series_uid, "MR", "t1_mprage_sag"
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = ROWS, COLS, 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    return ds


def _item(**kw):
    it = Dataset()
    for k, v in kw.items():
        setattr(it, k, v)
    return it


def write_classic(folder):
    uid = generate_uid()
    for k in range(N):
        ds = _base("1.2.840.10008.5.1.4.1.1.4", uid)
        ds.MRAcquisitionType, ds.ImageType = "3D", ["ORIGINAL", "PRIMARY", "M", "ND"]
        ds.ImageOrientationPatient = [*ROW_DIR, *COL_DIR]
        ds.ImagePositionPatient = list(P0 + k * STEP * NORMAL)
        ds.PixelSpacing, ds.SliceThickness, ds.RescaleSlope, ds.RescaleIntercept = PS, STEP, 2, -10
        ds.PixelData = TRUTH[k].tobytes()
        ds.save_as(folder / f"im{k:03d}.dcm", enforce_file_format=True)


def write_enhanced(path, stacks=(dict(),), geometry=True, seed=1):
    """One Enhanced MR file; each stack dict can set EffectiveEchoTime / ComplexImageComponent.
    Frames are stored in shuffled order, as scanners may do."""
    ds = _base("1.2.840.10008.5.1.4.1.1.4.1", generate_uid())
    ds.ImageType = ["ORIGINAL", "PRIMARY", "T1", "NONE"]
    ds.DimensionIndexSequence = Sequence([_item(DimensionIndexPointer=0x00209056), _item(DimensionIndexPointer=0x00209057)])
    ds.SharedFunctionalGroupsSequence = Sequence([_item(
        PixelMeasuresSequence=Sequence([_item(PixelSpacing=PS, SliceThickness=STEP)]),
        PlaneOrientationSequence=Sequence([_item(ImageOrientationPatient=[*ROW_DIR, *COL_DIR])]),
        PixelValueTransformationSequence=Sequence([_item(RescaleSlope=2, RescaleIntercept=-10, RescaleType="US")]),
        MRFOVGeometrySequence=Sequence([_item(MRAcquisitionType="3D")]),
        MRTimingAndRelatedParametersSequence=Sequence([_item(RepetitionTime=2300)]),
    )])
    frames = [(s, k) for s in range(len(stacks)) for k in range(N)]
    np.random.default_rng(seed).shuffle(frames)
    groups, pixels = [], []
    for s, k in frames:
        st = stacks[s]
        g = _item(FrameContentSequence=Sequence([_item(StackID=str(s + 1), InStackPositionNumber=k + 1,
                                                       DimensionIndexValues=[s + 1, k + 1])]),
                  MRImageFrameTypeSequence=Sequence([_item(FrameType=["ORIGINAL", "PRIMARY", "T1", "NONE"],
                                                           ComplexImageComponent=st.get("ComplexImageComponent", "MAGNITUDE"))]),
                  MREchoSequence=Sequence([_item(EffectiveEchoTime=st.get("EffectiveEchoTime", 2.3))]))
        if geometry:
            g.PlanePositionSequence = Sequence([_item(ImagePositionPatient=list(P0 + k * STEP * NORMAL))])
        groups.append(g)
        pixels.append(TRUTH[k] + s)
    ds.PerFrameFunctionalGroupsSequence = Sequence(groups)
    ds.NumberOfFrames = len(frames)
    ds.PixelData = np.stack(pixels).astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_enhanced_file_matches_classic_series(tmp_path):
    (tmp_path / "classic").mkdir(), (tmp_path / "enh").mkdir()
    write_classic(tmp_path / "classic")
    write_enhanced(tmp_path / "enh" / "IM0001")
    c, e = scan_folder(str(tmp_path / "classic")), scan_folder(str(tmp_path / "enh"))
    assert len(c) == len(e) == 1 and len(e[0].files) == N
    assert pick_t1(e) is e[0] and e[0].meta["MRAcquisitionType"] == "3D" and e[0].meta["voxel"] == [0.9, 0.8, 1.2]
    vc, ac = load_series(c[0].files)
    ve, ae = load_series(e[0].files)
    assert np.allclose(ae, ac) and np.array_equal(ve, vc)
    assert np.allclose(np.linalg.norm(ae[:3, :3], axis=0), [PS[1], PS[0], STEP])     # column, row, slice spacing
    assert np.allclose(ae[:3, 0] / PS[1], ROW_DIR) and np.allclose(ae[:3, 2] / STEP, NORMAL)
    assert np.allclose(ae[:3, 3], P0) and np.array_equal(ve, TRUTH * 2.0 - 10)       # slices in order, rescaled


def test_echoes_and_phase_become_separate_series(tmp_path):
    write_enhanced(tmp_path / "IM0001", stacks=(dict(EffectiveEchoTime=2.3), dict(EffectiveEchoTime=4.6),
                                                dict(ComplexImageComponent="PHASE")))
    found = scan_folder(str(tmp_path))
    assert len(found) == 3 and all(len(s.files) == N for s in found)
    assert sorted(s.meta["SeriesDescription"][-13:] for s in found) == [f"(part {i} of 3)" for i in (1, 2, 3)]
    assert [s.excluded for s in found].count("phase image") == 1
    for s in found:
        vol, aff = load_series(s.files)
        assert vol.shape == (N, ROWS, COLS) and np.allclose(aff[:3, 3], P0)
        assert len(np.unique(vol - (TRUTH * 2.0 - 10))) == 1    # one stack's frames only, in slice order


def test_multiframe_without_positions_is_rejected_with_a_plain_message(tmp_path):
    write_enhanced(tmp_path / "IM0001", geometry=False)
    (s,) = scan_folder(str(tmp_path))
    assert s.excluded == "multi-frame file without slice positions" and pick_t1([s]) is None
    with pytest.raises(RuntimeError, match="no slice positions"):
        load_series(s.files)
    reply, code = server.add_scan(str(tmp_path), str(tmp_path / "cases"))
    assert code == 400 and "multi-frame DICOM file BodyGlass can't read" in reply["error"]

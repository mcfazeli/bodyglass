"""Compressed multi-frame DICOM from public sample sets, one file per transfer syntax. The files are downloaded by
tools/fetch_compressed_samples.sh, which lists each source and licence and checks SHA-256 sums; no patient data
is added to the repository.

Frames are decoded with pylibjpeg, the decoder BodyGlass installs (requirements.txt), and compared with a
reference: an uncompressed copy of the same data where the sample set has one, otherwise a decode by a second
library (GDCM, test-only)."""
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.encaps import encapsulate, generate_frames
from pydicom.pixels import pixel_array
from pydicom.uid import UID, ExplicitVRLittleEndian

from bodyglass.dicom_scan import pick_t1, scan_folder, unsupported_syntax
from bodyglass.volume import load_series
from test_multiframe import write_enhanced

# Lossy colour JPEG decoders legitimately differ by a few levels per channel (here up to 3 of 255, where pylibjpeg
# and GDCM upsample half-width chroma differently on a sharp colour edge), so it gets 1.5 %; everything else 1 %.
LOSSY_COLOUR_JPEG_TOLERANCE = 0.015
LOSSY_TOLERANCE = 0.01
SAMPLES = Path(__file__).resolve().parents[1] / "sample_data" / "compressed_multiframe"
pytestmark = pytest.mark.skipif(not SAMPLES.is_dir(), reason="run tools/fetch_compressed_samples.sh first")


def j2k_lossy_two_frames(tmp_path):
    """No public multi-frame JPEG 2000 lossy file was found, so the real lossy codestreams of two CT images of the
    same size and format (D_CLUNIE_CT1_J2KI from GDCM's test data, 693_J2KI from pydicom's test files: 512x512,
    16-bit signed) are placed, unchanged, as the two frames of one file."""
    names = ("D_CLUNIE_CT1_J2KI.dcm", "693_J2KI.dcm")
    ds = pydicom.dcmread(SAMPLES / names[0])
    frames = [next(generate_frames(pydicom.dcmread(SAMPLES / n).PixelData, number_of_frames=1)) for n in names]
    ds.NumberOfFrames, ds.PixelData = 2, encapsulate(frames)
    ds["PixelData"].VR = "OB"
    ds.save_as(tmp_path / "CT_J2KI_2frame.dcm", enforce_file_format=True)
    return tmp_path / "CT_J2KI_2frame.dcm"


# (transfer syntax, file, uncompressed reference or None for GDCM, lossless)
CASES = {
    "jpeg-baseline": ("1.2.840.10008.1.2.4.50", "color3d_jpeg_baseline.dcm", None, False),
    "jpeg-lossless": ("1.2.840.10008.1.2.4.70", "EnhancedWithIPPPerFrameIOPShared.dcm", None, True),
    "j2k-lossless": ("1.2.840.10008.1.2.4.90", "emri_small_jpeg_2k_lossless.dcm", "emri_small.dcm", True),
    "j2k-lossy": ("1.2.840.10008.1.2.4.91", j2k_lossy_two_frames, None, False),
}


@pytest.mark.parametrize("case", CASES)
def test_every_frame_decodes_and_matches_the_reference(case, tmp_path):
    uid, src, ref, lossless = CASES[case]
    path = src(tmp_path) if callable(src) else SAMPLES / src
    ds = pydicom.dcmread(path)
    assert ds.file_meta.TransferSyntaxUID == uid and unsupported_syntax(ds) is None
    px = pixel_array(path, decoding_plugin="pylibjpeg", raw=True).astype(np.int64)   # raw: colour stays YBR
    n = int(ds.NumberOfFrames)
    assert n > 1 and px.shape[0] == n
    want = pydicom.dcmread(SAMPLES / ref).pixel_array if ref else \
        pixel_array(path, decoding_plugin="gdcm", raw=True)
    diff = np.abs(px - want.astype(np.int64))
    if lossless:
        assert diff.max() == 0
        return
    colour_jpeg = uid == "1.2.840.10008.1.2.4.50" and ds.SamplesPerPixel == 3
    tolerance = LOSSY_COLOUR_JPEG_TOLERANCE if colour_jpeg else LOSSY_TOLERANCE
    assert diff.max() <= tolerance * (2 ** int(ds.BitsStored) - 1)   # share of the stored value range


def test_jpeg_lossless_enhanced_file_builds_a_volume(tmp_path):
    """The JPEG Lossless sample is an enhanced multi-frame file with per-frame positions, so it goes through the
    whole scan-and-load path. It is listed as 'not MR' (it is a 3D X-ray), so it is loaded directly."""
    shutil.copy(SAMPLES / "EnhancedWithIPPPerFrameIOPShared.dcm", tmp_path)
    (s,) = scan_folder(str(tmp_path))
    assert len(s.files) == 189 and s.excluded == "not MR"
    vol, _ = load_series(s.files)
    assert vol.shape == (189, 183, 183)


def test_unsupported_compression_is_listed_not_used(tmp_path):
    uid, name = "1.2.840.10008.1.2.4.102", "MPEG-4 AVC/H.264"   # video: no still-image decoder reads it
    p = tmp_path / "IM0001"
    write_enhanced(p)
    ds = pydicom.dcmread(p)
    ds.file_meta.TransferSyntaxUID = UID(uid)
    ds.PixelData = encapsulate([b"\x00" * 8] * int(ds.NumberOfFrames))   # placeholder frames
    ds["PixelData"].VR = "OB"
    ds.save_as(p, enforce_file_format=True)
    (s,) = scan_folder(str(tmp_path))
    assert s.files and pick_t1([s]) is None
    assert name in s.excluded and uid in s.excluded
    with pytest.raises(RuntimeError, match=uid.replace(".", r"\.")):
        load_series(s.files)


def test_uncompressed_files_are_never_flagged(tmp_path):
    write_enhanced(tmp_path / "IM0001")
    assert pydicom.dcmread(tmp_path / "IM0001").file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    (s,) = scan_folder(str(tmp_path))
    assert not s.meta["unsupported"]

"""Build a TEST study folder that mimics the layout of a typical patient-portal export of a Siemens MRI exam.

The only real images are the public sample MPRAGE (open-dicom/siemens_mprage_0).
All other series are SYNTHETIC decoys resampled from that same public T1 volume, with
headers set like FLAIR / T2 / SWI / DWI / ADC / MPR series, so the metadata-based series
detection can be exercised. Folder names are deliberately meaningless (S0001...).
"""
import os, random, shutil, sys
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bodyglass.volume import load_series, to_ras_grid  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "sample_data/siemens_mprage_0/dcm"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sample_data/test_study"

files = sorted(os.path.join(SRC, f) for f in os.listdir(SRC) if f.endswith(".dcm"))
vol, aff = load_series(files)
grid, origin, sp = to_ras_grid(vol, aff, 1.0)
tmpl = pydicom.dcmread(files[0])
study_uid = tmpl.StudyInstanceUID

# desc, n, acq, scanseq, seqname, imagetype, thickness, inplane, rows, cols, orientation
DECOYS = [
    ("T1 SAG 3D_MPR_Axial", 55, "3D", ["GR", "IR"], "*tfl3d1_16", ["DERIVED", "SECONDARY", "MPR"], 3.0, 1.0, 256, 256, "ax"),
    ("AX T2 FLAIR", 30, "2D", ["SE", "IR"], "*tir2d1_16", ["ORIGINAL", "PRIMARY", "M", "NORM"], 5.0, 0.9, 256, 232, "ax"),
    ("AX T2 TSE", 30, "2D", ["SE"], "*tse2d1_15", ["ORIGINAL", "PRIMARY", "M", "NORM"], 5.0, 0.45, 512, 464, "ax"),
    ("SWI_Images", 128, "3D", ["GR"], "*swi3d1r", ["ORIGINAL", "PRIMARY", "SWI", "NORM"], 1.5, 0.9, 256, 224, "ax"),
    ("Mag_Images", 128, "3D", ["GR"], "*swi3d1r", ["ORIGINAL", "PRIMARY", "M", "NORM"], 1.5, 0.9, 256, 224, "ax"),
    ("Pha_Images", 128, "3D", ["GR"], "*swi3d1r", ["ORIGINAL", "PRIMARY", "P", "ND"], 1.5, 0.9, 256, 224, "ax"),
    ("mIP_Images(SW)", 121, "3D", ["GR"], "*swi3d1r", ["DERIVED", "PRIMARY", "MNIP", "NORM"], 1.5, 0.9, 256, 224, "ax"),
    ("AX DIFFUSION_TRACEW", 30, "2D", ["EP"], "*ep_b1000t", ["ORIGINAL", "PRIMARY", "DIFFUSION", "TRACEW", "ND"], 5.0, 1.2, 192, 192, "ax"),
    ("AX DIFFUSION_TRACEW", 30, "2D", ["EP"], "*ep_b0", ["ORIGINAL", "PRIMARY", "DIFFUSION", "TRACEW", "ND"], 5.0, 1.2, 192, 192, "ax"),
    ("AX DIFFUSION_ADC", 30, "2D", ["EP"], "*ep_b0_1000", ["DERIVED", "PRIMARY", "DIFFUSION", "ADC", "ND"], 5.0, 1.2, 192, 192, "ax"),
]


def sample_slice(ipp, row_dir, col_dir, ps, rows, cols):
    r, c = np.mgrid[0:rows, 0:cols]
    lps = ipp[None, None] + c[..., None] * ps * row_dir + r[..., None] * ps * col_dir
    ras = lps * np.array([-1, -1, 1])
    idx = (ras - origin) / sp
    return ndi.map_coordinates(grid, [idx[..., 2], idx[..., 1], idx[..., 0]], order=1, cval=0)


if os.path.exists(OUT):
    shutil.rmtree(OUT)
order = list(range(1, len(DECOYS) + 2))
random.Random(7).shuffle(order)
# 1) the real public MPRAGE, copied verbatim
d = os.path.join(OUT, "DICOM", f"S{order[0]:04d}")
os.makedirs(d)
for i, f in enumerate(files):
    shutil.copy(f, os.path.join(d, f"IM{i:05d}"))
# 2) synthetic decoys
center = np.array([float(x) for x in (origin + np.array(grid.shape[::-1]) * sp / 2)]) * np.array([-1, -1, 1])  # LPS
for k, (desc, n, acq, seq, seqname, itype, th, ps, rows, cols, _o) in enumerate(DECOYS):
    d = os.path.join(OUT, "DICOM", f"S{order[k + 1]:04d}")
    os.makedirs(d)
    suid = generate_uid()
    row_dir, col_dir, normal = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    start = center - row_dir * ps * cols / 2 - col_dir * ps * rows / 2 - normal * th * (n - 1) / 2
    for i in range(n):
        ipp = start + normal * th * i
        img = sample_slice(ipp, row_dir, col_dir, ps, rows, cols)
        ds = tmpl.copy()
        ds.SeriesInstanceUID = suid
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = desc; ds.ProtocolName = desc
        ds.SeriesNumber = 10 + k; ds.InstanceNumber = i + 1
        ds.MRAcquisitionType = acq; ds.ScanningSequence = seq; ds.SequenceName = seqname
        ds.ImageType = itype
        ds.SliceThickness = th; ds.SpacingBetweenSlices = th
        ds.PixelSpacing = [ps, ps]; ds.Rows = rows; ds.Columns = cols
        ds.ImageOrientationPatient = [*row_dir, *col_dir]
        ds.ImagePositionPatient = [float(x) for x in ipp]
        ds.SliceLocation = float(ipp[2])
        for tag in ("InversionTime",):
            if "IR" not in seq and tag in ds:
                del ds[tag]
        ds.PixelData = np.clip(img, 0, 4095).astype(np.uint16).tobytes()
        ds.save_as(os.path.join(d, f"IM{i:05d}"))
print("wrote", OUT, "with", len(DECOYS) + 1, "series")

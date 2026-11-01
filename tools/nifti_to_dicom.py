"""Convert a public NIfTI T1 into a classic single-frame MR DICOM series (TEST DATA ONLY).

Slices are written along the first voxel axis (sagittal for these BIDS files), with
ImagePositionPatient / ImageOrientationPatient derived from the NIfTI affine, so the
viewer's DICOM loader and metadata-based series detection are exercised end to end.
Header values that are not in the NIfTI (sequence, TI...) are set to plausible 3D T1 values
and are marked synthetic in ImageComments.
"""
import argparse, os
import numpy as np
import nibabel as nib
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

ap = argparse.ArgumentParser()
ap.add_argument("nifti"); ap.add_argument("outdir")
ap.add_argument("--desc", default="T1 SAG 3D"); ap.add_argument("--manufacturer", default="")
ap.add_argument("--source", default="")
a = ap.parse_args()

img = nib.load(a.nifti)
data = np.asarray(img.dataobj, dtype=np.float32)
if data.ndim == 4:
    data = data[..., 0]
A = img.affine
lo, hi = np.percentile(data, [0, 99.9])
data = np.clip((data - lo) / (hi - lo), 0, 1.2) * 3000  # -> 0..3600 uint16
ni, nj, nk = data.shape
LPS = np.diag([-1.0, -1.0, 1.0])
cj, ck = LPS @ A[:3, 1], LPS @ A[:3, 2]
row_dir = cj / np.linalg.norm(cj)            # +column index  <-> +j
col_dir = -ck / np.linalg.norm(ck)           # +row index     <-> -k (superior at top)
ps = [float(np.linalg.norm(ck)), float(np.linalg.norm(cj))]
th = float(np.linalg.norm(A[:3, 0]))
os.makedirs(a.outdir, exist_ok=True)
study, series, frame = generate_uid(), generate_uid(), generate_uid()
for i in range(ni):
    ipp = LPS @ (A @ np.array([i, 0, nk - 1, 1.0]))[:3]
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = generate_uid(); ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
    ds.Modality = "MR"; ds.Manufacturer = a.manufacturer
    ds.PatientName = "Public^Sample"; ds.PatientID = "PUBLIC"
    ds.SeriesDescription = a.desc; ds.ProtocolName = a.desc; ds.SeriesNumber = 3; ds.InstanceNumber = i + 1
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.MRAcquisitionType = "3D"; ds.ScanningSequence = ["GR", "IR"]; ds.SequenceName = "*tfl3d1"
    ds.SliceThickness = th; ds.SpacingBetweenSlices = th; ds.PixelSpacing = ps
    ds.ImageOrientationPatient = [float(x) for x in (*row_dir, *col_dir)]
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.Rows, ds.Columns = nk, nj
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    ds.ImageComments = f"SYNTHETIC DICOM converted from public NIfTI: {a.source}"
    sl = data[i, :, ::-1].T  # rows = -k, cols = j
    ds.PixelData = np.ascontiguousarray(sl).astype(np.uint16).tobytes()
    ds.save_as(os.path.join(a.outdir, f"IM{i:05d}"), enforce_file_format=True)
print(f"wrote {ni} slices {nk}x{nj}, spacing {ps} x {th:.3g} -> {a.outdir}")

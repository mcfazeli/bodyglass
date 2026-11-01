"""Build a multi-series DICOM study from public BIDS NIfTI files (TEST DATA ONLY).

All series share one StudyInstanceUID and FrameOfReferenceUID (as on a real scanner), keep their
own NIfTI geometry (so the T1 and the other series start roughly aligned, exactly as in a real
export), and copy acquisition headers from the BIDS JSON sidecars (SeriesDescription, ImageType,
SequenceName, ScanningSequence, TE/TR/TI in ms, MRAcquisitionType...). DWI 4D data is turned into
the two images scanners usually export: a b=1000 TRACEW series and an ADC map (in 1e-6 mm^2/s).
Everything is marked SYNTHETIC in ImageComments.

usage: bids_to_dicom_study.py OUTDIR --source "ds004146 sub-0001 (CC0)" spec [spec ...]
  spec = NIFTI[:key=value,...]   keys: desc, number, kind=dwi|scalar, itype (ImageType with \\), slice=0|1|2 (voxel axis)
"""
import argparse, json, os
import numpy as np
import nibabel as nib
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

LPS = np.diag([-1.0, -1.0, 1.0])


def write_series(out, data, A, hdr, study, frame, number, source, extra=None):
    """data: 3D float array in NIfTI voxel order; A: NIfTI affine (RAS). One file per slice."""
    sp = np.linalg.norm(A[:3, :3], axis=0)
    K = int(hdr["slice"]) if "slice" in hdr else int(np.argmax(sp)) if sp.max() / sp.min() > 1.2 else 2   # slice axis: given, thickest (2D) or the 3rd
    R, C = [a for a in range(3) if a != K]
    dirs = [LPS @ A[:3, a] / sp[a] for a in range(3)]
    series = generate_uid()
    os.makedirs(out, exist_ok=True)
    lo, hi = np.percentile(data[np.isfinite(data)], [0.5, 99.8]) if hdr.get("_scale", True) else (0, 1)
    for s in range(data.shape[K]):
        idx = [slice(None)] * 3; idx[K] = s
        sl = data[tuple(idx)]                    # dims (R, C) in remaining-axis order
        if R > C:
            sl = sl.T
        vox0 = np.zeros(3); vox0[K] = s
        ipp = LPS @ (A @ np.r_[vox0, 1.0])[:3]
        ds = Dataset(); ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = generate_uid(); ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality = "MR"; ds.PatientName = "Public^Sample"; ds.PatientID = "PUBLIC"
        ds.StudyDescription = "BRAIN (public sample)"
        ds.Manufacturer = hdr.get("Manufacturer", ""); ds.ManufacturerModelName = hdr.get("ManufacturersModelName", "")
        ds.MagneticFieldStrength = hdr.get("MagneticFieldStrength", 3)
        ds.SeriesDescription = hdr.get("SeriesDescription", ""); ds.ProtocolName = hdr.get("ProtocolName", ds.SeriesDescription)
        ds.SeriesNumber = number; ds.InstanceNumber = s + 1
        ds.ImageType = hdr.get("ImageType", ["ORIGINAL", "PRIMARY", "M", "ND"])
        for k in ("MRAcquisitionType", "SequenceName"):
            if hdr.get(k): setattr(ds, k, hdr[k])
        for k in ("ScanningSequence", "SequenceVariant"):
            if hdr.get(k): setattr(ds, k, hdr[k] if isinstance(hdr[k], list) else str(hdr[k]).split("_"))
        for k in ("EchoTime", "RepetitionTime", "InversionTime"):
            if hdr.get(k) is not None: setattr(ds, k, round(float(hdr[k]) * 1000, 3))
        if hdr.get("FlipAngle") is not None: ds.FlipAngle = hdr["FlipAngle"]
        ds.SliceThickness = float(sp[K]); ds.SpacingBetweenSlices = float(sp[K])
        ds.PixelSpacing = [float(sp[R]), float(sp[C])]
        ds.ImageOrientationPatient = [float(x) for x in (*dirs[C], *dirs[R])]  # row dir (+col idx), col dir (+row idx)
        ds.ImagePositionPatient = [float(x) for x in ipp]
        ds.Rows, ds.Columns = sl.shape
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        for k, v in (extra or {}).items(): setattr(ds, k, v)
        ds.ImageComments = f"SYNTHETIC DICOM converted from public NIfTI: {source}"
        px = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1.2) * 3000 if hdr.get("_scale", True) else np.clip(sl, 0, 65535)
        ds.PixelData = np.ascontiguousarray(px).astype(np.uint16).tobytes()
        ds.save_as(os.path.join(out, f"IM{s:05d}"), enforce_file_format=True)
    return data.shape[K]


ap = argparse.ArgumentParser()
ap.add_argument("outdir"); ap.add_argument("--source", default=""); ap.add_argument("specs", nargs="+")
a = ap.parse_args()
study, frame = generate_uid(), generate_uid()
for n, spec in enumerate(a.specs, start=2):
    path, _, opts = spec.partition(":")
    o = dict(kv.split("=", 1) for kv in opts.split(",") if kv)
    js = path.replace(".nii.gz", ".json")
    hdr = json.load(open(js)) if os.path.exists(js) else {}
    if "desc" in o: hdr["SeriesDescription"] = hdr["ProtocolName"] = o["desc"]
    if "itype" in o: hdr["ImageType"] = o["itype"].split("\\")
    for k, v in o.items():   # any other key overrides a sidecar header (e.g. LEMON's sparse JSONs)
        if k not in ("desc", "number", "kind", "itype"):
            try: v = float(v)
            except ValueError: pass
            hdr[k] = v
    img = nib.load(path); A = img.affine
    number = int(o.get("number", n))
    if o.get("kind") == "dwi":
        d4 = np.asarray(img.dataobj, np.float32)
        bv = np.loadtxt(path.replace(".nii.gz", ".bval"))
        b0 = d4[..., bv < 50].mean(-1)
        sel = np.abs(bv - 1000) < 60
        trace = np.exp(np.log(np.maximum(d4[..., sel], 1e-3)).mean(-1))  # geometric mean = isotropic trace
        adc = np.where(b0 > 0.05 * np.percentile(b0, 99), -np.log(np.maximum(trace, 1e-3) / np.maximum(b0, 1e-3)) / 1000.0, 0)
        adc = np.clip(adc, 0, 0.004) * 1e6  # 1e-6 mm^2/s
        base = os.path.basename(path).split(".")[0]
        h1 = dict(hdr, SeriesDescription=o.get("desc", "ep2d_diff_b1000_TRACEW"), ImageType=["ORIGINAL", "PRIMARY", "DIFFUSION", "TRACEW", "ND"])
        h1["ProtocolName"] = h1["SeriesDescription"]
        k1 = write_series(os.path.join(a.outdir, f"S{number:02d}_TRACEW"), trace, A, h1, study, frame, number, a.source,
                          {"DiffusionBValue": 1000.0})
        h2 = dict(hdr, SeriesDescription=h1["SeriesDescription"].replace("TRACEW", "ADC"), ImageType=["DERIVED", "PRIMARY", "DIFFUSION", "ADC", "ND"], _scale=False)
        h2["ProtocolName"] = h1["ProtocolName"]
        k2 = write_series(os.path.join(a.outdir, f"S{number + 50:02d}_ADC"), adc, A, h2, study, frame, number + 50, a.source)
        print(f"{base}: TRACEW {k1} slices, ADC {k2} slices (median brain ADC {np.median(adc[adc > 200]):.0f}e-6 mm2/s)")
        continue
    data = np.asarray(img.dataobj, np.float32)
    if data.ndim == 4: data = data[..., 0]
    k = write_series(os.path.join(a.outdir, f"S{number:02d}"), data, A, hdr, study, frame, number, a.source)
    print(f"{os.path.basename(path)} -> S{number:02d} '{hdr.get('SeriesDescription','')}' {k} slices")

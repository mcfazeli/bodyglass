"""Find DICOM series in a folder and pick the high-res 3D T1 by metadata only.

Folder and file names are never used for the decision.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pydicom

HEADER_TAGS = [
    "SeriesInstanceUID", "SeriesDescription", "ProtocolName", "SequenceName",
    "ScanningSequence", "SequenceVariant", "MRAcquisitionType", "ImageType",
    "Modality", "PixelSpacing", "SliceThickness", "Rows", "Columns",
    "RepetitionTime", "EchoTime", "InversionTime", "SeriesNumber",
    "ImageOrientationPatient", "ImagePositionPatient", "Manufacturer",
    "DiffusionBValue", "FrameOfReferenceUID", "ImageComments",
    "BodyPartExamined", "StudyDescription", "StudyDate",   # Add scan routing (brain vs neck) and library dates
]


@dataclass
class Series:
    uid: str
    files: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    score: float = 0.0
    reasons: list = field(default_factory=list)
    excluded: str | None = None

    def summary(self) -> dict:
        m = self.meta
        return {
            "uid": self.uid,
            "description": m.get("SeriesDescription", ""),
            "seriesNumber": m.get("SeriesNumber"),
            "nFiles": len(self.files),
            "acquisition": m.get("MRAcquisitionType", ""),
            "imageType": "\\".join(m.get("ImageType", []) or []),
            "sequence": m.get("SequenceName", ""),
            "voxel": m.get("voxel"),
            "score": round(self.score, 1),
            "excluded": self.excluded,
            "reasons": self.reasons,
        }


def _val(ds, tag):
    v = getattr(ds, tag, None)
    if v is None:
        return None
    if isinstance(v, pydicom.multival.MultiValue):
        return [x if isinstance(x, str) else float(x) for x in v]
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
        return float(v)
    return v if isinstance(v, (int, float)) else str(v)


def scan_folder(root: str) -> list[Series]:
    """Read headers only (no pixel data) of every DICOM file under root.

    shortcut: one file = one slice; Enhanced multi-frame MR (one file per series, e.g. Siemens XA, newer
    Philips) is not read and its T1 is reported as 'not found'. Add frame splitting if a user's export needs it."""
    series: dict[str, Series] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True,
                                     specific_tags=HEADER_TAGS)
            except Exception:
                continue  # not DICOM (DICOMDIR viewers, html, etc.)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid or not hasattr(ds, "Rows"):
                continue
            s = series.setdefault(str(uid), Series(uid=str(uid)))
            s.files.append(path)
            if not s.meta:
                s.meta = {t: _val(ds, t) for t in HEADER_TAGS}
    for s in series.values():
        score_series(s)
    return sorted(series.values(), key=lambda s: -s.score)


def _text(m):
    parts = [m.get("SeriesDescription"), m.get("ProtocolName"), m.get("SequenceName")]
    return " ".join(str(p) for p in parts if p).upper()


def score_series(s: Series) -> None:
    m = s.meta
    text = _text(m)
    itype = [str(x).upper() for x in (m.get("ImageType") or [])]
    scan_seq = [str(x).upper() for x in (m.get("ScanningSequence") or [])] \
        if isinstance(m.get("ScanningSequence"), list) else [str(m.get("ScanningSequence") or "").upper()]
    n = len(s.files)
    ps = m.get("PixelSpacing") or [1.0, 1.0]
    st = float(m.get("SliceThickness") or 0) or None
    voxel = [round(float(ps[0]), 2), round(float(ps[1]), 2), round(st, 2) if st else None]
    m["voxel"] = voxel
    score, why = 0.0, []

    def add(pts, reason):
        nonlocal score
        score += pts
        why.append(f"{'+' if pts >= 0 else ''}{pts:g} {reason}")

    if (m.get("Modality") or "MR") != "MR":
        s.excluded = "not MR"
    tokens = set(re.split(r"[^A-Z0-9*]+", text))
    hard = [
        (("SWI", "SWI_", "MIP", "MINIP", "MNIP", "PHA", "SW", "SWAN"), "SWI / susceptibility"),
        (("DWI", "DIFFUSION", "TRACE", "TRACEW", "ADC", "DTI", "FA"), "diffusion / ADC"),
        (("FLAIR",), "FLAIR"),
        (("T2", "TSE", "T2W", "PD"), "T2 / PD-weighted"),
        (("LOCALIZER", "SCOUT", "LOC", "SURVEY"), "localizer"),
    ]
    if "SWI" in text or "SWAN" in text:
        s.excluded = "SWI / susceptibility"
    if "P" in itype or "PHASE" in itype:
        s.excluded = s.excluded or "phase image"
    for keys, name in hard:
        hit = any(k in tokens for k in keys) or any(k in itype for k in keys)
        if hit and not s.excluded:
            s.excluded = name
    if "EP" in scan_seq and not s.excluded:
        s.excluded = "echo-planar (fMRI/DWI)"

    if m.get("MRAcquisitionType") == "3D":
        add(30, "MRAcquisitionType=3D")
    elif m.get("MRAcquisitionType") == "2D":
        add(-20, "2D acquisition")
    if any(k in text for k in ("T1", "MPRAGE", "MP-RAGE", "MP_RAGE", "BRAVO", "TFE", "TFL", "SPGR", "FSPGR")):
        add(25, "description/sequence says T1/MPRAGE")
    if "GR" in scan_seq:
        add(10, "gradient-echo sequence (GR)")
    if "IR" in scan_seq:
        add(5, "inversion recovery (MPRAGE-like)")
    ti = m.get("InversionTime")
    if ti and 600 <= float(ti) <= 1300:
        add(5, f"TI={float(ti):g} ms typical of MPRAGE")
    if "DERIVED" in itype or "SECONDARY" in itype or "MPR" in tokens or "REFORMAT" in text:
        add(-40, "derived/reformatted (MPR) image")
    if "ORIGINAL" in itype:
        add(5, "ORIGINAL image")
    # MP2RAGE exports INV1, INV2, UNI and a T1 map with the same sequence; only UNI is T1-weighted (and is DERIVED)
    if "UNI" in itype or "UNI" in tokens:
        add(45, "MP2RAGE UNI image (the T1-weighted one)")
    elif tokens & {"INV1", "INV2"}:
        add(-50, "MP2RAGE inversion image (low contrast)")
    elif "T1 MAP" in itype or tokens & {"T1MAP", "T1MAPS"} or ("MP2RAGE" in text and "T1" in tokens):
        add(-30, "quantitative T1 map")
    add(min(n, 256) / 8.0, f"{n} slices")
    if st:
        vs = [float(ps[0]), float(ps[1]), st]
        aniso = max(vs) / min(vs)
        if aniso <= 1.3 and max(vs) <= 1.3:
            add(20, f"near-isotropic {vs[0]:.2g}x{vs[1]:.2g}x{vs[2]:.2g} mm")
        elif max(vs) > 3:
            add(-15, f"thick slices ({max(vs):.2g} mm)")
    if n < 60:
        add(-25, "too few slices for a 3D model")
    s.score = score - (1000 if s.excluded else 0)
    s.reasons = why + ([f"excluded: {s.excluded}"] if s.excluded else [])


def pick_t1(series: list[Series]) -> Series | None:
    ok = [s for s in series if not s.excluded and len(s.files) >= 60]
    if ok:
        return max(ok, key=lambda s: s.score)
    return None


# ---------------------------------------------------------------- other sequences
KIND_ORDER = ["flair", "t2", "tse", "swi", "mip", "swi_mag", "dwi", "adc"]


def classify_kind(s: Series) -> tuple[str | None, str]:
    """Which non-T1 sequence a series is, by metadata only. Returns (kind or None, reason)."""
    m = s.meta
    text = _text(m)
    tokens = set(re.split(r"[^A-Z0-9]+", text))
    itype = [str(x).upper() for x in (m.get("ImageType") or [])]
    ss = m.get("ScanningSequence")
    scan_seq = [str(x).upper() for x in ss] if isinstance(ss, list) else str(ss or "").upper().split("\\")
    seqname = str(m.get("SequenceName") or "").upper()
    bval = m.get("DiffusionBValue")
    if (m.get("Modality") or "MR") != "MR":
        return None, "not MR"
    if tokens & {"LOCALIZER", "SCOUT", "LOC", "SURVEY", "AAHEAD"}:
        return None, "localizer"
    if "ADC" in itype or "ADC" in tokens:
        return "adc", "ImageType/description says ADC"
    if "TRACEW" in itype or "TRACE" in itype or (bval and float(bval) > 0) \
            or ("DIFFUSION" in itype and "EP" in scan_seq) or tokens & {"DWI", "TRACEW", "TRACE"}:
        if "FA" in itype or "FA" in tokens or "COLFA" in tokens:
            return None, "diffusion FA map"
        return "dwi", "diffusion-weighted (ImageType / b-value)"
    swi_family = "SWI" in seqname or "SWI" in text or "SWAN" in text or "SWI" in itype
    if "MNIP" in itype or "MIP" in itype or tokens & {"MIP", "MINIP", "MNIP"}:
        return ("mip" if swi_family or "GR" in scan_seq else None), "minimum-intensity projection"
    if swi_family:
        if "P" in itype or "PHASE" in itype or tokens & {"PHA", "PHASE"}:
            return None, "SWI phase (not shown: hard to read)"
        if "SWI" in itype or "SWI" in tokens:
            return "swi", "susceptibility-weighted (ImageType SWI)"
        return "swi_mag", "SWI magnitude echo"
    ti = float(m.get("InversionTime") or 0)
    if "FLAIR" in tokens or "TIRM" in text or "DARK" in text or "DA-FL" in text or "DAFL" in text \
            or ("IR" in scan_seq and "SE" in scan_seq and ti > 1500):
        return "flair", "FLAIR (dark-fluid inversion recovery)"
    if "EP" in scan_seq:
        return None, "echo-planar (fMRI)"
    if tokens & {"T2", "T2W", "TSE", "SPC", "SPACE", "CUBE", "VISTA"} or "TSE" in seqname or "SPC" in seqname:
        if "PD" in tokens:
            return None, "proton density"
        if m.get("MRAcquisitionType") == "2D" or "TSE" in tokens or "TSE" in seqname:
            return "tse", "T2 turbo spin echo (2D)"
        return "t2", "T2-weighted (3D)"
    return None, "not a recognised sequence"


def _coverage(s: Series) -> float:
    st = float(s.meta.get("SliceThickness") or 1.0)
    return len(s.files) * st


def pick_sequences(series: list[Series], t1: Series | None) -> dict:
    """Best series per kind. Returns {kind: {"series": Series, "reason": str, "alternatives": n}}."""
    groups: dict[str, list] = {}
    for s in series:
        if t1 is not None and s.uid == t1.uid:
            continue
        kind, why = classify_kind(s)
        s.meta["kind"], s.meta["kindReason"] = kind, why
        if kind:
            groups.setdefault(kind, []).append(s)
    out = {}
    for kind, lst in groups.items():
        def rank(s):
            itype = [str(x).upper() for x in (s.meta.get("ImageType") or [])]
            return ((s.meta.get("MRAcquisitionType") == "3D") * 30 + min(_coverage(s), 180) / 3
                    + ("ORIGINAL" in itype) * 5 + min(len(s.files), 200) / 20)
        best = max(lst, key=rank)
        out[kind] = {"series": best, "reason": best.meta["kindReason"],
                     "alternatives": len(lst) - 1}
    # A 3D T2 makes a partial 2D TSE redundant only if the user has no TSE; keep both (they look different).
    return {k: out[k] for k in KIND_ORDER if k in out}

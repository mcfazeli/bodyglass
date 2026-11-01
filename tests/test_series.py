"""Series picking and classification use DICOM headers only; these mimic real exports."""
from bodyglass.dicom_scan import Series, classify_kind, pick_t1, score_series

T1 = dict(Modality="MR", MRAcquisitionType="3D", SequenceName="tfl3d1_16", ScanningSequence=["GR", "IR"],
          PixelSpacing=[1.0, 1.0], SliceThickness=1.0, ImageType=["ORIGINAL", "PRIMARY", "M", "ND"])


def ser(desc, n=176, **meta):
    s = Series(uid=desc, files=[f"f{i}" for i in range(n)], meta=dict(T1, SeriesDescription=desc, **meta))
    score_series(s)
    return s


def test_mp2rage_picks_uni_not_inversions_or_t1_map():
    lst = [ser("mp2rage_INV1", InversionTime=700.0), ser("mp2rage_INV2", InversionTime=2500.0),
           ser("mp2rage_UNI_Images", ImageType=["DERIVED", "PRIMARY", "M", "UNI", "ND"]),
           ser("mp2rage_T1_Images", ImageType=["DERIVED", "PRIMARY", "T1 MAP", "ND"])]
    assert pick_t1(lst).uid == "mp2rage_UNI_Images"


def test_original_mprage_beats_mpr_reformat_and_decoys():
    lst = [ser("MPRAGE", InversionTime=900.0),
           ser("T1 SAG 3D_MPR_Axial", ImageType=["DERIVED", "SECONDARY", "MPR"]),
           ser("t2_spc_da-fl_sag", SequenceName="spcir_278"), ser("SWI_Images", SequenceName="swi3d1r"),
           ser("AAHead_Scout", n=128), ser("T1 thin slab", n=40)]
    assert pick_t1(lst).uid == "MPRAGE"
    assert {s.uid for s in lst if s.excluded} >= {"t2_spc_da-fl_sag", "SWI_Images"}


def test_nothing_usable_returns_none():
    assert pick_t1([ser("ep2d_bold", ScanningSequence=["EP"]), ser("T1 short", n=30)]) is None


def k(desc, **meta):
    return classify_kind(Series(uid=desc, files=["a"] * 30, meta=dict(T1, SeriesDescription=desc, **meta)))[0]


def test_classify_kinds():
    assert k("t2_tirm_tra_dark-fluid", MRAcquisitionType="2D", ScanningSequence=["SE", "IR"]) == "flair"
    assert k("t2_tse_tra", MRAcquisitionType="2D", ScanningSequence=["SE"]) == "tse"
    assert k("t2_spc_sag", SequenceName="spc_314", ScanningSequence=["SE"]) == "t2"
    assert k("SWI_Images", ImageType=["ORIGINAL", "PRIMARY", "M", "SWI"]) == "swi"
    assert k("Pha_Images", SequenceName="swi3d1r", ImageType=["ORIGINAL", "PRIMARY", "P", "ND"]) is None
    assert k("mIP_Images(SW)", SequenceName="swi3d1r", ImageType=["ORIGINAL", "PRIMARY", "MNIP"]) == "mip"
    assert k("ep2d_diff_b1000_TRACEW", ImageType=["ORIGINAL", "PRIMARY", "DIFFUSION", "TRACEW"],
             ScanningSequence=["EP"]) == "dwi"
    assert k("ep2d_diff_b1000_ADC", ImageType=["DERIVED", "PRIMARY", "DIFFUSION", "ADC"]) == "adc"
    assert k("localizer") is None and k("MPRAGE") is None

"""Neck exam: series pick (metadata only), landmark fit (C1-C3 shapes), label merge."""
import numpy as np
from scipy.spatial.transform import Rotation

from bodyglass.dicom_scan import Series
from bodyglass.neck import fit_rigid, merge_labels, pick_neck

SAG, AX = [0, 1, 0, 0, 0, -1], [1, 0, 0, 0, 1, 0]


def ser(desc, n, iop, acq, thick, px=0.8, **meta):
    return Series(uid=desc, files=[f"f{i}" for i in range(n)], meta=dict(
        Modality="MR", SeriesDescription=desc, ImageOrientationPatient=iop, MRAcquisitionType=acq,
        SliceThickness=thick, PixelSpacing=[px, px], ImageType=["ORIGINAL", "PRIMARY", "M"], **meta))


def test_pick_prefers_sagittal_3d_even_with_unhelpful_names():
    series = [ser("Series 7", 64, SAG, "3D", 0.9),                      # a 3D SPACE with a meaningless name
              ser("T2 TSE AX", 30, AX, "2D", 3.5, SequenceName="tse2d1_15"),
              ser("T2 TSE SAG", 15, SAG, "2D", 3.0, SequenceName="tse2d1_17"),   # too few images
              ser("STIR SAG", 17, SAG, "2D", 3.0),
              ser("localizer", 9, SAG, "2D", 7.0),
              ser("ep2d_diff", 40, AX, "2D", 4.0, ScanningSequence="EP")]
    assert pick_neck(series).uid == "Series 7"
    assert pick_neck(series[1:]).uid == "T2 TSE AX"                     # nothing better: still usable
    assert pick_neck(series[4:]) is None                                # localizer / diffusion only


def vertebra(z, y0, dens=False):
    """1 mm points of a vertebra-like shape: body in front, ring + spinous process behind (asymmetric)."""
    g = np.mgrid[-26:27, -26:27, -6:7].reshape(3, -1).T.astype(float)
    x, y, zz = g.T
    body = (x / 9) ** 2 + ((y - 11) / 7) ** 2 + (zz / 5) ** 2 <= 1
    r = np.hypot(x * 0.8, y)
    ring = (r > 8) & (r < 13) & (y < 4) & (abs(zz) < 3)
    spine = (abs(x) < 2) & (y < -12) & (y > -24 + z / 10) & (abs(zz - 1) < 3)
    keep = body | ring | spine
    if dens:
        keep |= (np.hypot(x, y - 11) < 4) & (zz > 4)
    return g[keep] + [0, y0, z]


def test_landmark_fit_recovers_turn_about_the_spine_from_partial_views():
    # centroids nearly on one line, the brain scan ends inside C3 and the neck exam is a 40 mm slab:
    # centroids alone are off by ~2 mm here; the shape refinement on the shared parts is exact
    whole = {"C1": vertebra(60, 3), "C2": vertebra(42, 3), "C3": vertebra(25, 3)}
    fix = {k: v[v[:, 2] > 24] for k, v in whole.items()}
    R = Rotation.from_euler("zxy", [15, 6, -4], degrees=True).as_matrix()   # 15 deg about the long axis
    t = np.array([12.0, -30.0, 25.0])
    to_neck = lambda P: (P - t) @ R                                     # brain mm -> neck exam mm
    mov = {k: to_neck(v) for k, v in whole.items()}
    mov = {k: v[np.abs(v[:, 0]) < 20] for k, v in mov.items()}
    T, rms = fit_rigid(mov, fix, lambda P: np.abs(P[:, 0]) < 20, lambda P: P[:, 2] > 24)
    probe = np.array([[0, 10, -80.0], [0, 0, 60], [10, -20, 0]])         # incl. a point 100 mm below C3
    err = np.linalg.norm((to_neck(probe) @ T[:3, :3].T + T[:3, 3]) - probe, axis=1)
    assert err.max() < 1.0, err
    assert rms < 1.0


def test_landmark_fit_reports_shape_mismatch():
    fix = {"C1": vertebra(60, 2), "C2": vertebra(42, 4, dens=True), "C3": vertebra(25, 3)}
    other = {"C1": vertebra(66, 9) * 1.3, "C2": vertebra(40, -3) * 0.7, "C3": vertebra(18, 6) * 1.25}   # not the same spine
    _T, rms = fit_rigid(other, fix, lambda P: np.ones(len(P), bool), lambda P: np.ones(len(P), bool))
    assert rms > 3.0


def test_merge_keeps_brain_labels():
    brain = np.array([0, 5, 5, 0, 0], np.uint8)
    neck = np.array([209, 209, 0, 200, 0], np.int16)
    assert merge_labels(brain, neck).tolist() == [209, 5, 5, 200, 0]


def test_add_scan_routing_brain_neck_both():
    from bodyglass.dicom_scan import score_series
    from bodyglass.neck import route
    t1 = ser("t1_mprage_sag", 176, SAG, "3D", 1.0, SequenceName="tfl3d1")
    brain_t2 = ser("t2_space_sag", 176, SAG, "3D", 1.0, SequenceName="spc")          # a brain exam's own sagittal T2
    neck_t2 = ser("Series 7", 64, SAG, "3D", 0.9, BodyPartExamined="CSPINE")
    for s in (t1, brain_t2, neck_t2):
        score_series(s)
    r = route([t1, brain_t2])
    assert r["brain"] is t1 and r["neck"] is None              # brain exam only: its T2 is not a "neck"
    r = route([t1, brain_t2, neck_t2])
    assert r["brain"] is t1 and r["neck"] is neck_t2           # whole export with both exams
    unlabelled = ser("T2 TSE SAG", 40, SAG, "2D", 3.0)
    score_series(unlabelled)
    r = route([unlabelled])
    assert r["brain"] is None and r["neck"] is unlabelled      # no brain T1: any series may be the neck
    head_neck = ser("T1w head-neck", 192, SAG, "3D", 1.0, SequenceName="tfl3d1")
    score_series(head_neck)
    assert route([head_neck, unlabelled])["brain"] is head_neck  # "head-neck" T1 is a head scan, not a spine one

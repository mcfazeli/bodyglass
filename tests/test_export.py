"""3D-print export: one watertight body, life size in mm, valid 3MF."""
import io
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh

from bodyglass.export3d import build_solid, write_3mf


def test_overlapping_parts_become_one_watertight_solid_in_mm():
    a = trimesh.creation.icosphere(4, radius=20.0)
    b = trimesh.creation.icosphere(4, radius=15.0)
    b.apply_translation((25.0, 0, 0))
    mesh, pitch = build_solid([a, b])
    d, r1, r2 = 25.0, 20.0, 15.0   # analytic union volume of two spheres
    lens = np.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)
    union = 4 / 3 * np.pi * (r1 ** 3 + r2 ** 3) - lens
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
    assert abs(mesh.volume - union) / union < 0.05, (mesh.volume, union)
    assert np.allclose(mesh.bounds, [[-20, -20, -20], [40, 20, 20]], atol=1.5)


def test_label_mask_keeps_its_volume():
    k = np.zeros((40, 40, 40), bool)
    k[10:30, 10:30, 10:30] = True                    # 20 mm cube on a 1 mm grid = 8 mL
    mesh, _ = build_solid([], masks=[(k, np.array([-20.0, -20.0, -20.0]), 1.0)])
    assert mesh.is_watertight and abs(mesh.volume - 8000) / 8000 < 0.08


def test_tiny_selection_does_not_vanish():
    a = trimesh.creation.icosphere(3, radius=1.2)   # two bodies of ~250 faces each
    b = a.copy(); b.apply_translation((30.0, 0, 0))
    mesh, _ = build_solid([a, b])
    assert len(mesh.faces) > 0 and mesh.is_watertight


def test_3mf_is_millimetres_and_complete():
    m = trimesh.creation.box((10, 20, 30))
    z = zipfile.ZipFile(io.BytesIO(write_3mf(m, "t")))
    assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= set(z.namelist())
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert root.get("unit") == "millimeter"
    assert len(root.findall(".//m:vertex", ns)) == len(m.vertices)
    assert len(root.findall(".//m:triangle", ns)) == len(m.faces)

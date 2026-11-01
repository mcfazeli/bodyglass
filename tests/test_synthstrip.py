"""The slabbed SynthStrip forward pass must equal the plain (upstream mri_synthstrip) forward pass."""
import pytest

torch = pytest.importorskip("torch")
from bodyglass import synthstrip  # noqa: E402


def upstream_forward(model, x):   # StripModel.forward from freesurfer/mri_synthstrip, unchanged logic
    hist = [x]
    for convs in model.encoder:
        for conv in convs:
            x = conv(x)
        hist.append(x)
        x = torch.nn.functional.max_pool3d(x, 2)
    for convs in model.decoder:
        for conv in convs:
            x = conv(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, hist.pop()], 1)
    for conv in model.remaining:
        x = conv(x)
    return x


def test_slabbed_forward_is_exact():
    torch.manual_seed(0)
    model = synthstrip._build(torch)().eval()
    x = torch.rand(1, 1, 64, 64, 64)
    with torch.inference_mode():
        a, b = synthstrip._forward(model, x), upstream_forward(model, x)
    assert a.shape == b.shape == (1, 1, 64, 64, 64)
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())

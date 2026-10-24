"""Exercise the dependency APIs covered by the security-version overrides."""
import io
import ssl

import certifi
import numpy as np
import pytest
from fastembed.image.transform.operators import Compose, ResizeForVisionEncoder
from filelock import FileLock, Timeout
from huggingface_hub.utils import WeakFileLock
from PIL import Image


@pytest.mark.parametrize("mode,color,expected", [
    ("RGBA", (255, 0, 0, 255), (1, -1, -1)),
    ("L", 128, (1 / 255, 1 / 255, 1 / 255)),
])
def test_fastembed_decodes_resizes_and_normalizes_with_pillow(mode, color, expected):
    encoded = io.BytesIO()
    Image.new(mode, (48, 32), color).save(encoded, format="PNG")
    encoded.seek(0)
    processor = Compose.from_config({
        "do_resize": True,
        "size": {"shortest_edge": 24},
        "do_center_crop": True,
        "crop_size": {"height": 16, "width": 16},
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    })
    with Image.open(encoded) as image:
        result = processor([image])[0]
    assert result.shape == (3, 16, 16)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0, 0], expected, atol=1e-6)


def test_fastembed_vision_resize_preserves_pixel_channels():
    pixels = np.zeros((3, 12, 24), dtype=np.uint8)
    pixels[0] = 255
    resized = ResizeForVisionEncoder(max_size=16)([pixels])[0]
    assert resized.shape == (3, 16, 32)
    assert np.isfinite(resized).all()
    np.testing.assert_array_equal(resized[:, 0, 0], [255, 0, 0])


def test_huggingface_filelock_contention_and_release(tmp_path):
    path = tmp_path / "model-cache.lock"
    with WeakFileLock(path):
        with pytest.raises(Timeout):
            with FileLock(path, timeout=0):
                pytest.fail("A second cache writer acquired the held lock")
    with FileLock(path, timeout=0) as acquired:
        assert acquired.is_locked


def test_certifi_bundle_loads_into_tls_context():
    context = ssl.create_default_context(cafile=certifi.where())
    assert context.cert_store_stats()["x509_ca"] > 0
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname

from __future__ import annotations

import numpy as np

from clockd.services.view_transformer import ViewTransformer
from clockd.utils.units import convert_speed


def test_convert_speed_mph():
    assert convert_speed(10.0, "mph") == 22.4


def test_convert_speed_kmh():
    assert convert_speed(10.0, "kmh") == 36.0


def test_view_transformer_pipeline():
    source = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
    target = np.array([[0, 0], [8, 0], [8, 40], [0, 40]], dtype=np.float32)
    vt = ViewTransformer(source, target)

    # Bottom-center of a detection at (320, 480) should map to (4, 40)
    result = vt.transform_points(np.array([[320, 480]]))
    np.testing.assert_allclose(result, [[4.0, 40.0]], atol=0.1)

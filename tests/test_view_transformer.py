from __future__ import annotations

import numpy as np
import pytest

from clockd.services.view_transformer import ViewTransformer


def test_identity_transform():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    vt = ViewTransformer(pts, pts)
    result = vt.transform_points(np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(result, [[0.5, 0.5]], atol=1e-5)


def test_scaling_transform():
    source = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    target = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    vt = ViewTransformer(source, target)
    result = vt.transform_points(np.array([[50, 50]]))
    np.testing.assert_allclose(result, [[5.0, 5.0]], atol=1e-5)


def test_empty_points():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    vt = ViewTransformer(source, source)
    result = vt.transform_points(np.empty((0, 2)))
    assert result.shape == (0, 2)


def test_invalid_shape():
    with pytest.raises(ValueError, match="must each be"):
        ViewTransformer(np.zeros((3, 2)), np.zeros((4, 2)))

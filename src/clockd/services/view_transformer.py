from __future__ import annotations

import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != (4, 2) or target.shape != (4, 2):
            raise ValueError("source and target must each be (4, 2) arrays")
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        transformed = cv2.perspectiveTransform(pts, self.m)
        return transformed.reshape(-1, 2)

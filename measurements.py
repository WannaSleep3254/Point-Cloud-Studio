"""Float64 measurements in unchanged source coordinate units (no unit inference)."""
from dataclasses import dataclass

import numpy as np


# Reject unresolved length-measurement pairs relative to coordinate precision.
DUPLICATE_RTOL = 1e-7


@dataclass(frozen=True)
class CircleMeasurement:
    center: np.ndarray
    radius: float
    normal: np.ndarray

    @property
    def diameter(self):
        return 2.0 * self.radius


def validate_measurement_points(points):
    """Reject nonfinite coordinates and coincident / numerically close pairs."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not 1 <= len(p) <= 2:
        raise ValueError("XYZ 좌표를 가진 점 1~2개가 필요합니다.")
    if not np.isfinite(p).all():
        raise ValueError("유한한 좌표의 점을 선택하세요.")
    if len(p) > 1:
        edges = np.array([p[j] - p[i] for i in range(len(p))
                          for j in range(i + 1, len(p))])
        scale = float(np.max(np.abs(edges)))
        if not np.isfinite(scale):
            raise ValueError("좌표 범위가 너무 커 계산할 수 없습니다.")
        if scale == 0:
            raise ValueError("이미 선택한 점과 같습니다. 다른 점을 선택하세요.")
        lengths = np.linalg.norm(edges / scale, axis=1)
        # Absolute precision guard follows coordinate magnitude, not a fixed unit.
        resolution = float(np.max(np.abs(p))) * (32 * np.finfo(np.float64).eps)
        if lengths.min() <= max(DUPLICATE_RTOL * lengths.max(), resolution / scale):
            raise ValueError("선택점이 중복되거나 너무 가깝습니다. 더 떨어진 점을 선택하세요.")
    return p

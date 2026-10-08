"""Find dominant approximate planes, without changing source coordinates.

RANSAC on a deterministic bounded sample, least-squares refinement, then assign
ALL displayed finite points to the nearest accepted plane within the tolerance.
Coplanar disconnected regions may belong to one result. Residual statistics
describe this fit; they are not minimum-zone flatness or a certified tolerance.
"""
from dataclasses import dataclass

import numpy as np


class SurfaceDetectionCancelled(Exception):
    pass


@dataclass
class DetectedPlane:
    center: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    bounds_uv: np.ndarray
    indices: np.ndarray
    rms: float
    max_deviation: float
    residual_span: float
    tolerance: float
    sample_count: int
    finite_count: int

    @property
    def fraction(self):
        return len(self.indices) / self.finite_count


def _fit_plane(points):
    center = points.mean(axis=0)
    delta = points - center
    values, vectors = np.linalg.eigh(delta.T @ delta / len(points))
    if values[2] <= 0 or values[1] <= values[2] * 1e-5:
        raise ValueError('점들이 일직선에 가까워 평면을 정할 수 없습니다.')
    normal = vectors[:, 0]
    # A plane has two equivalent normal signs. Choose a stable convention.
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    u = vectors[:, 2]
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    basis = np.column_stack((u, np.cross(normal, u)))
    return center, normal, basis, values


def detect_planes(xyz, tolerance=0.0, min_fraction=0.05, max_planes=5,
                  progress=None, cancelled=None):
    """Extract planes; tolerance in source units (0=0.2% of sampled max extent).

    min_fraction is relative to the original finite display population, not the
    progressively shrinking remainder. Plane memberships do not overlap.
    """
    report = progress or (lambda _: None)

    def check():
        if cancelled and cancelled():
            raise SurfaceDetectionCancelled()

    if (not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(min_fraction)
            or not 0.01 <= min_fraction <= 1 or not isinstance(max_planes, (int, np.integer))
            or not 1 <= max_planes <= 10):
        raise ValueError('허용 오차는 0 이상, 최소 비율은 1~100%, 최대 평면 수는 1~10이어야 합니다.')
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 30:
        raise ValueError('평면 검출에는 표시점이 최소 30개 필요합니다.')
    check()
    report(0)
    # Keep indices in the original displayed array for highlighting/export use.
    finite_parts = []
    for start in range(0, len(xyz), 100_000):
        check()
        finite_parts.append(start + np.flatnonzero(np.isfinite(xyz[start:start + 100_000]).all(axis=1)))
    finite_indices = np.concatenate(finite_parts)
    if len(finite_indices) < 30:
        raise ValueError('평면 검출에 필요한 유한 좌표가 부족합니다.')
    rng = np.random.default_rng(2107)
    sampled_indices = rng.choice(finite_indices, min(len(finite_indices), 120_000), replace=False)
    sample = np.array(xyz[sampled_indices], dtype=np.float64)
    base = np.median(sample, axis=0)
    scale = float(np.max(np.ptp(sample, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('점들이 겹쳐 있어 평면을 정할 수 없습니다.')
    sample = (sample - base) / scale
    tol = tolerance / scale if tolerance else 0.002
    if not 1e-12 <= tol <= 0.05:
        raise ValueError('허용 오차는 분석 범위의 5% 이하이며 수치적으로 구분 가능한 값이어야 합니다.')
    minimum = max(30, int(np.ceil(min_fraction * len(sample))))
    available = np.ones(len(sample), dtype=bool)
    models = []
    report(5)
    for plane_number in range(max_planes):
        check()
        remaining = np.flatnonzero(available)
        if len(remaining) < minimum:
            break
        subset = sample[rng.choice(remaining, min(8000, len(remaining)), replace=False)]
        best_count, best_model = 0, None
        trial_limit = 2000
        trial = 0
        while trial < trial_limit:
            if trial % 50 == 0:
                check()
                report(5 + int(65 * (plane_number + trial / 2000) / max_planes))
            a, b, c = subset[rng.choice(len(subset), 3, replace=False)]
            u, v = b - a, c - a
            normal = np.cross(u, v)
            size = np.linalg.norm(normal)
            trial += 1
            if size <= 1e-5 * max(u @ u, v @ v):
                continue
            normal /= size
            count = np.count_nonzero(np.abs((subset - a) @ normal) <= tol)
            if count > best_count:
                best_count, best_model = int(count), (a, normal)
                fraction = count / len(subset)
                if fraction > 0.05:
                    confidence_trials = int(np.ceil(np.log(0.001) / np.log(max(1e-12, 1 - fraction ** 3))))
                    trial_limit = min(trial_limit, max(150, confidence_trials))
        if best_model is None:
            break
        point, normal = best_model
        active_points = sample[remaining]
        mask = np.abs((active_points - point) @ normal) <= tol
        if mask.sum() < minimum:
            break
        try:
            for _ in range(4):
                check()
                center, normal, basis, values = _fit_plane(active_points[mask])
                refined = np.abs((active_points - center) @ normal) <= tol
                if refined.sum() < minimum:
                    raise ValueError('평면 지지점이 부족합니다.')
                if np.array_equal(mask, refined):
                    break
                mask = refined
            center, normal, basis, values = _fit_plane(active_points[mask])
            mask = np.abs((active_points - center) @ normal) <= tol
            # Reject lines, very narrow strips, and slabs of volumetric noise.
            if (mask.sum() < minimum or np.sqrt(values[1]) < tol * 3
                    or values[0] > values[1] * 0.02):
                raise ValueError('안정적인 평면 지지를 찾지 못했습니다.')
            duplicate = any(abs(normal @ old_normal) >= np.cos(np.radians(2))
                            and abs((center - old_center) @ old_normal) <= 2 * tol
                            for old_center, old_normal, _ in models)
            if not duplicate:
                models.append((center, normal, basis))
        except (ValueError, np.linalg.LinAlgError):
            # Do not repeatedly propose the same degenerate slab.
            pass
        available[remaining[mask]] = False
    if not models:
        raise ValueError('주요 평면을 찾지 못했습니다. 허용 오차·최소 점 비율 또는 Z 범위를 조정하세요.')

    # Exclusive nearest-plane assignment on every finite displayed point.
    minimum_full = max(30, int(np.ceil(min_fraction * len(finite_indices))))
    first_assignment = True
    while models:
        labels = np.full(len(xyz), -1, dtype=np.int8)
        centers = np.array([m[0] for m in models])
        normals = np.array([m[1] for m in models])
        offsets = np.einsum('ij,ij->i', centers, normals)
        for start in range(0, len(finite_indices), 100_000):
            check()
            ids = finite_indices[start:start + 100_000]
            p = (np.asarray(xyz[ids], dtype=np.float64) - base) / scale
            distances = np.abs(p @ normals.T - offsets)
            nearest = distances.argmin(axis=1)
            accepted = distances[np.arange(len(p)), nearest] <= tol
            labels[ids[accepted]] = nearest[accepted]
            if first_assignment:
                report(70 + int(15 * min(start + len(ids), len(finite_indices)) / len(finite_indices)))
        counts = np.bincount(labels[labels >= 0], minlength=len(models))
        keep = counts >= minimum_full
        if keep.all():
            break
        # A discarded weak plane must not steal points from retained planes.
        models = [model for model, accepted in zip(models, keep) if accepted]
        first_assignment = False

    results = []
    for number, (center, normal, basis) in enumerate(models):
        check()
        ids = np.flatnonzero(labels == number)
        if len(ids) < minimum_full:
            continue
        sum_sq = 0.0
        low, high = np.inf, -np.inf
        bounds = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
        for start in range(0, len(ids), 100_000):
            check()
            delta = (np.asarray(xyz[ids[start:start + 100_000]], dtype=np.float64) - base) / scale - center
            residual = delta @ normal
            sum_sq += float(residual @ residual)
            low, high = min(low, residual.min()), max(high, residual.max())
            uv = delta @ basis
            bounds[0] = np.minimum(bounds[0], uv.min(axis=0))
            bounds[1] = np.maximum(bounds[1], uv.max(axis=0))
        results.append(DetectedPlane(base + center * scale, normal, basis, bounds * scale, ids,
                                     float(np.sqrt(sum_sq / len(ids)) * scale),
                                     float(max(abs(low), abs(high)) * scale), float((high - low) * scale),
                                     tol * scale, len(sample), len(finite_indices)))
        report(85 + int(14 * (number + 1) / len(models)))
    if not results:
        raise ValueError('최소 점 비율을 만족하는 평면이 없습니다. 최소 비율 또는 허용 오차를 조정하세요.')
    results.sort(key=lambda result: (-len(result.indices), result.rms))
    check()
    report(100)
    return results

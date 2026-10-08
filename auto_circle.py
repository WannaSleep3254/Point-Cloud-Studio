"""Automatic circular boundary detection on a dominant, approximately planar surface.

Bounded deterministic sampling -> PCA plane -> occupied-grid boundary -> circle
RANSAC -> geometric least-squares refinement in the fitted 3D plane. Results are
estimates of boundary circles, not metrology certification or cylinder fitting.
"""
from dataclasses import dataclass

import numpy as np

from measurements import CircleMeasurement


class DetectionCancelled(Exception):
    pass


@dataclass
class DetectedCircle:
    circle: CircleMeasurement
    support: np.ndarray
    rms: float
    arc_degrees: float
    tolerance: float
    sample_count: int
    boundary_count: int


def _neighbors(mask, operation):
    padded = np.pad(mask, 1)
    views = [padded[i:i + mask.shape[0], j:j + mask.shape[1]] for i in range(3) for j in range(3)]
    return operation(views, axis=0)


def _boundary_points(points, uv, cell):
    index = np.floor((uv - uv.min(axis=0)) / cell).astype(int) + 3
    shape = tuple(index.max(axis=0) + 4)
    keys = np.ravel_multi_index(index.T, shape)
    counts = np.bincount(keys, minlength=int(np.prod(shape)))
    occupied = counts.reshape(shape) > 0
    # Close gaps of one grid cell. Keep actual points near the resulting edge.
    closed = _neighbors(_neighbors(occupied, np.any), np.all)
    border = closed & ~_neighbors(closed, np.all)
    wanted = (_neighbors(border, np.any) & occupied).ravel()
    cells = np.flatnonzero(wanted)
    result = np.column_stack([np.bincount(keys, weights=points[:, i], minlength=len(counts))[cells]
                              / counts[cells] for i in range(3)])
    return result


def _plane(points):
    origin = points.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh((points - origin).T @ (points - origin))
    return origin, vectors[:, [2, 1]], vectors[:, 0], eigenvalues


def _fit2d(points):
    mean = points.mean(axis=0)
    q = points - mean
    solution, _, rank, _ = np.linalg.lstsq(np.column_stack((2 * q, np.ones(len(q)))),
                                         np.einsum('ij,ij->i', q, q), rcond=None)
    if rank < 3:
        raise ValueError('원 맞춤에 충분한 곡률이 없습니다.')
    center = solution[:2] + mean
    radius = float(np.sqrt(max(0, solution[2] + solution[:2] @ solution[:2])))
    for _ in range(8):
        delta = points - center
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances < 1e-14) or radius <= 0:
            raise ValueError('원 맞춤이 불안정합니다.')
        jacobian = np.column_stack((-delta / distances[:, None], -np.ones(len(points))))
        step = np.linalg.lstsq(jacobian, radius - distances, rcond=None)[0]
        center += step[:2]
        radius += step[2]
        if np.linalg.norm(step) < 1e-10:
            break
    return center, radius


def _arc(points, center):
    angles = np.sort(np.mod(np.arctan2(*(points - center)[:, ::-1].T), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    degrees = float(np.degrees(2 * np.pi - gaps.max()))
    bins = np.unique(np.floor(angles * 72 / (2 * np.pi)).astype(int))
    continuity = len(bins) / min(72, max(1, int(np.ceil(degrees / 5)) + 1))
    return degrees, continuity


def detect_circles(xyz, tolerance=0.0, min_diameter=0.0, max_diameter=0.0,
                   min_arc=120.0, max_candidates=5, progress=None, cancelled=None):
    """Return ranked circle candidates; tolerance/diameters use source units, 0=auto."""
    report = progress or (lambda _: None)

    def check():
        if cancelled and cancelled():
            raise DetectionCancelled()

    if min(tolerance, min_diameter, max_diameter) < 0 or not 30 <= min_arc <= 360:
        raise ValueError('허용 오차/지름은 0 이상, 최소 호 범위는 30~360도여야 합니다.')
    if max_diameter and max_diameter <= min_diameter:
        raise ValueError('최대 지름은 최소 지름보다 커야 합니다.')
    p = np.asarray(xyz)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 30:
        raise ValueError('자동 검출에는 표시된 점이 최소 30개 필요합니다.')
    check()
    rng = np.random.default_rng(2107)
    indices = rng.choice(len(p), min(len(p), 200_000), replace=False)
    p = np.array(p[indices], dtype=np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 30:
        raise ValueError('자동 검출에 필요한 유한 좌표가 부족합니다.')
    base = np.median(p, axis=0)
    scale = float(np.max(np.ptp(p, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('점들이 겹쳐 있어 윤곽을 검출할 수 없습니다.')
    p = (p - base) / scale
    origin, basis, normal, values = _plane(p)
    if values[1] <= values[2] * 1e-6 or values[0] > values[1] * 0.25:
        raise ValueError('평면에 가까운 원형 윤곽을 찾기 어렵습니다. Z 범위로 대상 표면을 좁혀 주세요.')
    uv = (p - origin) @ basis
    cell = float(np.max(np.ptp(uv, axis=0)) / 256)
    tol = tolerance / scale if tolerance else cell * 1.6
    if tol <= np.finfo(float).eps or tol > np.max(np.ptp(uv, axis=0)) * 0.1:
        raise ValueError('허용 오차가 형상 크기에 비해 너무 작거나 큽니다.')
    boundary = _boundary_points(p, uv, cell)
    if len(boundary) > 4000:
        boundary = boundary[rng.choice(len(boundary), 4000, replace=False)]
    if len(boundary) < 24:
        raise ValueError('원형 윤곽점이 부족합니다. 표본 수 또는 Z 범위를 조정하세요.')
    boundary_uv = (boundary - origin) @ basis
    available = np.ones(len(boundary), dtype=bool)
    min_radius = max(min_diameter / scale / 2, cell * 6)
    max_radius = max_diameter / scale / 2 if max_diameter else np.max(np.ptp(uv, axis=0)) * 2
    results = []
    report(10)
    for candidate in range(max_candidates):
        check()
        active = np.flatnonzero(available)
        if len(active) < 24:
            break
        best_score, best_mask = 0, None
        coords = boundary_uv[active]
        for iteration in range(1800):
            if iteration % 50 == 0:
                check()
                report(min(94, 10 + int(84 * (candidate + iteration / 1800) / max_candidates)))
            a, b, c = coords[rng.choice(len(coords), 3, replace=False)]
            u, v = b - a, c - a
            cross = u[0] * v[1] - u[1] * v[0]
            if abs(cross) < 1e-7:
                continue
            center = a + np.array([v[1] * (u @ u) - u[1] * (v @ v),
                                   u[0] * (v @ v) - v[0] * (u @ u)]) / (2 * cross)
            radius = np.linalg.norm(a - center)
            if not min_radius <= radius <= max_radius:
                continue
            error = abs(np.linalg.norm(coords - center, axis=1) - radius)
            mask = error <= tol
            n = int(mask.sum())
            if n < 24 or n < best_score:
                continue
            arc, continuity = _arc(coords[mask], center)
            if arc < min_arc or continuity < 0.72:
                continue
            score = n * continuity
            if score > best_score:
                best_score, best_mask = score, mask
        if best_mask is None:
            break
        support_indices = active[best_mask]
        # Fit the actual edge points in 3D, then reselect by spatial residual.
        try:
            for _ in range(3):
                selected = boundary[support_indices]
                local_origin, local_basis, local_normal, _ = _plane(selected)
                projected = (selected - local_origin) @ local_basis
                center2, radius = _fit2d(projected)
                center3 = local_origin + local_basis @ center2
                delta = boundary[active] - center3
                height = delta @ local_normal
                radial = np.linalg.norm(delta - height[:, None] * local_normal, axis=1) - radius
                spatial_error = np.hypot(height, radial)
                support_indices = active[spatial_error <= tol * 1.5]
                if len(support_indices) < 24:
                    raise ValueError('평면 원 지지점이 부족합니다.')
            selected = boundary[support_indices]
            local_origin, local_basis, local_normal, _ = _plane(selected)
            projected = (selected - local_origin) @ local_basis
            center2, radius = _fit2d(projected)
            center3 = local_origin + local_basis @ center2
            arc, continuity = _arc(projected, center2)
            delta = selected - center3
            height = delta @ local_normal
            radial = np.linalg.norm(delta - height[:, None] * local_normal, axis=1) - radius
            rms = float(np.sqrt(np.mean(height ** 2 + radial ** 2)))
            if (arc < min_arc or continuity < 0.72 or rms > tol or not min_radius <= radius <= max_radius
                    or rms / radius > 0.025):
                raise ValueError('적합도 기준을 충족하지 못했습니다.')
            circle = CircleMeasurement(base + scale * center3, float(radius * scale), local_normal)
            if not any(np.linalg.norm(r.circle.center - circle.center) < 3 * tol * scale
                       and abs(r.circle.radius - circle.radius) < 3 * tol * scale for r in results):
                results.append(DetectedCircle(circle, base + scale * selected, rms * scale,
                                              arc, tol * scale, len(p), len(boundary)))
        except (ValueError, np.linalg.LinAlgError):
            pass
        # Remove this hypothesis even if it fails the 3D check, then try others.
        available[active[best_mask]] = False
    check()
    if not results:
        raise ValueError('적합도 기준을 만족하는 원을 찾지 못했습니다. Z 범위·지름 범위·허용 오차를 조정하세요.')
    results.sort(key=lambda r: (-r.arc_degrees * len(r.support), r.rms))
    report(100)
    return results

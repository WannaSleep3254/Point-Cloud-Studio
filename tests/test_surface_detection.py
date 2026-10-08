import unittest

import numpy as np

from surface_detection import SurfaceDetectionCancelled, detect_planes


def plane(normal, center, count=4000, noise=0.002, seed=12):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-5, 5, (count, 2))
    return center + uv[:, :1] * u + uv[:, 1:] * v + rng.normal(0, noise, (count, 1)) * normal


class SurfaceTests(unittest.TestCase):
    def test_horizontal_vertical_tilted_planes(self):
        for normal in ([0, 0, 1], [1, 0, 0], [1, 2, -3]):
            with self.subTest(normal=normal):
                expected = np.array(normal, dtype=float); expected /= np.linalg.norm(expected)
                points = plane(normal, np.array([2, -3, 7]))
                result = detect_planes(points, tolerance=0.02)[0]
                self.assertGreater(abs(result.normal @ expected), 0.999999)
                self.assertLess(abs((result.center - [2, -3, 7]) @ expected), 0.002)
                self.assertEqual(len(result.indices), len(points))
                residuals = (points[result.indices] - result.center) @ result.normal
                self.assertAlmostEqual(result.rms, np.sqrt(np.mean(residuals ** 2)), places=12)
                self.assertAlmostEqual(result.residual_span, np.ptp(residuals), places=12)
                self.assertLess(result.max_deviation, result.tolerance)

    def test_multiple_planes_and_outliers_have_exclusive_membership(self):
        points = np.vstack([plane([0, 0, 1], np.array([0, 0, 0]), count=6000),
                            plane([1, 0, 0], np.array([9, 0, 0]), count=4000),
                            plane([0, 1, 1], np.array([0, 15, 15]), count=3000),
                            np.random.default_rng(6).uniform(-20, 20, (2000, 3))])
        results = detect_planes(points, tolerance=0.02, min_fraction=0.1)
        self.assertEqual(len(results), 3)
        self.assertEqual([len(r.indices) for r in results], sorted((len(r.indices) for r in results), reverse=True))
        all_ids = np.concatenate([r.indices for r in results])
        self.assertEqual(len(all_ids), len(np.unique(all_ids)))
        for result in results:
            self.assertGreaterEqual(result.fraction, 0.1)
            self.assertLessEqual(np.max(abs((points[result.indices] - result.center) @ result.normal)), 0.020001)

    def test_parallel_planes_and_maximum_count(self):
        points = np.vstack([plane([0, 0, 1], np.array([0, 0, 0])),
                            plane([0, 0, 1], np.array([0, 0, 2]), count=2500)])
        self.assertEqual(len(detect_planes(points, tolerance=0.02)), 2)
        self.assertEqual(len(detect_planes(points, tolerance=0.02, max_planes=1)), 1)

    def test_noise_layers_do_not_duplicate_one_plane(self):
        points = plane([0, 0, 1], np.zeros(3), count=10000, noise=0.02)
        results = detect_planes(points, tolerance=0.02)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0].indices), 6000)

    def test_unit_scale_and_large_translation(self):
        base_points = plane([1, 2, 3], np.zeros(3), noise=0)
        for scale, offset in [(1e-8, np.zeros(3)), (1e8, np.array([1e12, -1e12, 1e12]))]:
            result = detect_planes(base_points * scale + offset, tolerance=0.02 * scale)[0]
            self.assertEqual(len(result.indices), len(base_points))
            self.assertLess(result.rms / scale, 1e-8)

    def test_nonfinite_indices_map_to_original_data(self):
        points = np.vstack([np.full((2, 3), np.nan), plane([0, 0, 1], np.zeros(3))])
        result = detect_planes(points, tolerance=0.02)[0]
        self.assertEqual(result.finite_count, len(points) - 2)
        np.testing.assert_array_equal(result.indices, np.arange(2, len(points)))

    def test_full_membership_beyond_analysis_sample(self):
        points = plane([0, 1, 0], np.zeros(3), count=130000, noise=0)
        result = detect_planes(points)[0]
        self.assertEqual(result.sample_count, 120000)
        self.assertEqual(len(result.indices), 130000)

    def test_degenerate_and_unstructured_clouds_rejected(self):
        rng = np.random.default_rng(7)
        for points in (np.zeros((100, 3)), np.column_stack([np.arange(100), np.zeros((100, 2))]),
                       rng.uniform(-1, 1, (15000, 3))):
            with self.assertRaises(ValueError):
                detect_planes(points)

    def test_cancel_progress_and_invalid_options(self):
        points = plane([0, 0, 1], np.zeros(3))
        with self.assertRaises(SurfaceDetectionCancelled):
            detect_planes(points, cancelled=lambda: True)
        progress = []
        detect_planes(points, progress=progress.append)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)
        for options in ({'tolerance': -1}, {'tolerance': np.nan}, {'min_fraction': 0}, {'max_planes': 0}):
            with self.assertRaises(ValueError):
                detect_planes(points, **options)


if __name__ == '__main__':
    unittest.main()

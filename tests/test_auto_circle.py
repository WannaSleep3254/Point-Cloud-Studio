import unittest
import numpy as np

from auto_circle import DetectionCancelled, detect_circles


def ring(inner=9, outer=10, arc=360, seed=1, center=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.radians(arc), 45000)
    radius = rng.uniform(inner, outer, len(angle))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                            rng.normal(0, 0.008, len(angle)))) + center


class AutoCircleTests(unittest.TestCase):
    def test_inner_and_outer_boundaries(self):
        results = detect_circles(ring(), min_arc=300)
        diameters = [r.circle.diameter for r in results]
        self.assertTrue(any(abs(d - 18) < 0.3 for d in diameters), diameters)
        self.assertTrue(any(abs(d - 20) < 0.3 for d in diameters), diameters)
        self.assertTrue(all(r.rms < r.tolerance for r in results))

    def test_tilted_and_translated_partial_circle(self):
        points = ring(9.9, 10, arc=180)
        u = np.array([1, 2, 3]) / np.sqrt(14)
        v = np.cross(u, [0, 0, 1]); v /= np.linalg.norm(v)
        n = np.cross(u, v)
        center = np.array([100, -50, 25])
        points = points @ np.array([u, v, n]) + center
        results = detect_circles(points)
        result = min(results, key=lambda r: abs(r.circle.diameter - 20))
        self.assertLess(abs(result.circle.diameter - 19.9), 0.3)
        self.assertLess(np.linalg.norm(result.circle.center - center), 0.2)
        self.assertGreater(abs(result.circle.normal @ n), 0.999)
        self.assertGreater(result.arc_degrees, 160)

    def test_vertical_circle_and_units(self):
        points = ring(9.95, 10)[:, [2, 1, 0]] * 0.001
        result = detect_circles(points)[0]
        self.assertLess(abs(result.circle.diameter - 0.01995), 0.0003)

    def test_diameter_bounds(self):
        results = detect_circles(ring(), min_diameter=19.5, max_diameter=20.5)
        self.assertTrue(all(19.5 <= r.circle.diameter <= 20.5 for r in results))
        with self.assertRaises(ValueError):
            detect_circles(ring(), min_diameter=50, max_diameter=60)

    def test_line_rectangle_and_volume_rejected(self):
        rng = np.random.default_rng(8)
        examples = [np.column_stack((np.arange(100), np.zeros((100, 2)))),
                    np.column_stack((rng.uniform(-10, 10, (60000, 2)), np.zeros(60000))),
                    rng.normal(size=(10000, 3))]
        for points in examples:
            with self.assertRaises(ValueError):
                detect_circles(points)

    def test_cancellation_and_invalid_options(self):
        with self.assertRaises(DetectionCancelled):
            detect_circles(ring(), cancelled=lambda: True)
        with self.assertRaises(ValueError):
            detect_circles(ring(), min_diameter=20, max_diameter=10)


if __name__ == '__main__':
    unittest.main()

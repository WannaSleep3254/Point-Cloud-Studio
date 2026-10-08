"""Length point validation; manual three-point circle fitting has been removed."""
import unittest
import numpy as np
from measurements import validate_measurement_points


class LengthTests(unittest.TestCase):
    def test_points_are_float64_without_changing_coordinates(self):
        source = np.array([[1, 2, 3], [4, 6, 3]], dtype=np.float32)
        result = validate_measurement_points(source)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(np.linalg.norm(result[1] - result[0]), 5)
        np.testing.assert_array_equal(result, source)

    def test_duplicate_and_unresolved_points_rejected(self):
        for points in ([[[3, 4, 5], [3, 4, 5]], [[1e12, 0, 0], [1e12 + 0.001, 0, 0]]]):
            with self.assertRaises(ValueError):
                validate_measurement_points(points)

    def test_invalid_coordinates_and_point_counts(self):
        for points in ([], [[1, 2]], np.eye(3), [[0, np.nan, 0]], [[0, np.inf, 0]]):
            with self.assertRaises(ValueError):
                validate_measurement_points(points)

    def test_small_units_preserved(self):
        result = validate_measurement_points([[0, 0, 0], [1e-20, 0, 0]])
        self.assertEqual(result[1, 0], 1e-20)


if __name__ == '__main__':
    unittest.main()

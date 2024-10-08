import unittest

from normalization import normalize_to_0_1, normalize_to_bounds


class TestCustomNormalization(unittest.TestCase):
    def normalize_to_0_1(self):
        test_x = normalize_to_0_1([-10, 0, 10])
        self.assertTrue(test_x[0] == 0)
        self.assertTrue(test_x[1] == 0.5)
        self.assertTrue(test_x[2] == 1)
        return

    def normalize_to_bounds(self):
        test_x = normalize_to_bounds([-10, 0, 10], bounds=1)
        self.assertTrue(test_x[0] == -1)
        self.assertTrue(test_x[1] == 0)
        self.assertTrue(test_x[2] == 1)
        test_x = normalize_to_bounds([-10, 0, 10], bounds=100)
        self.assertTrue(test_x[0] == -100)
        self.assertTrue(test_x[1] == 0)
        self.assertTrue(test_x[2] == 100)
        return


if __name__ == "__main__":
    unittest.main()

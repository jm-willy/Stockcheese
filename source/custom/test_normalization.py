import unittest

from normalization import normalize_to_0_1, normalize_to_bounds


class TestCustomNormalization(unittest.TestCase):
    def normalize_to_0_1(self):
        test_x = normalize_to_0_1([-10, 0, 10])
        self.assertTrue(test_x[0] == 0)
        self.assertTrue(test_x[1] == 0.5)
        self.assertTrue(test_x[2] == 1)
        test_x = normalize_to_bounds([1, 1, 1], bounds=1, norm_same=True)
        self.assertTrue(test_x[0] == 0.5)
        self.assertTrue(test_x[1] == 0.5)
        self.assertTrue(test_x[2] == 0.5)
        test_x = normalize_to_bounds([1, 1, 1], bounds=1, norm_same=False)
        self.assertTrue(test_x[0] == 1)
        self.assertTrue(test_x[1] == 1)
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
        test_x = normalize_to_bounds([1, 1, 1], bounds=1, norm_same=True)
        self.assertTrue(test_x[0] == 0)
        self.assertTrue(test_x[1] == 0)
        self.assertTrue(test_x[2] == 0)
        test_x = normalize_to_bounds([1, 1, 1], bounds=1, norm_same=False)
        self.assertTrue(test_x[0] == 1)
        self.assertTrue(test_x[1] == 1)
        self.assertTrue(test_x[2] == 1)
        return


if __name__ == "__main__":
    unittest.main()

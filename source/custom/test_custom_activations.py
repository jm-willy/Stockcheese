import unittest

from Stockcheese.source.custom.activations import (
    my_hard_sigmoid,
    my_hard_tanh,
    leaky_hard_tanh,
)


class TestCustomLoss(unittest.TestCase):
    def test_my_hard_sigmoid(self):
        self.assertTrue(my_hard_sigmoid(0.0) == 0)
        self.assertTrue(my_hard_sigmoid(1.0, f_range=2) == 0.5)
        self.assertTrue(my_hard_sigmoid(1.0, f_range=4) == 0.25)
        self.assertTrue(my_hard_sigmoid(2.0, f_range=2) == 1)
        self.assertTrue(my_hard_sigmoid(4.0, f_range=4) == 1)
        return

    def test_my_hard_tanh(self):
        self.assertTrue(my_hard_tanh(0.0) == 0)
        self.assertTrue(my_hard_tanh(-2.0, f_range=2) == -1)
        self.assertTrue(my_hard_tanh(-4.0, f_range=4) == -1)
        self.assertTrue(my_hard_tanh(1.0, f_range=2) == 0.5)
        self.assertTrue(my_hard_tanh(1.0, f_range=4) == 0.25)
        self.assertTrue(my_hard_tanh(2.0, f_range=2) == 1)
        self.assertTrue(my_hard_tanh(4.0, f_range=4) == 1)
        return

    def test_leaky_hard_tanh(self):
        self.assertTrue(leaky_hard_tanh(0.0) == 0)
        self.assertTrue(leaky_hard_tanh(-2.0, f_range=2, slope=0.05) == -1.05)
        self.assertTrue(leaky_hard_tanh(-4.0, f_range=4, slope=0.05) == -1.05)
        self.assertTrue(leaky_hard_tanh(2.0, f_range=2, slope=0.05) == 1.05)
        self.assertTrue(leaky_hard_tanh(4.0, f_range=4, slope=0.05) == 1.05)
        return


if __name__ == "__main__":
    unittest.main()

import unittest

from activations import (
    my_hard_sigmoid,
    my_hard_tanh,
    leaky_hard_tanh,
    proportional_repr,
)


class TestCustomLoss(unittest.TestCase):
    def test_my_hard_sigmoid(self):
        self.assertTrue(my_hard_sigmoid(0.0) == 0)
        self.assertTrue(my_hard_sigmoid(0.5) == 0.5)
        self.assertTrue(my_hard_sigmoid(1.0) == 1)
        self.assertTrue(my_hard_sigmoid(2.0) == 1)
        return

    def test_my_hard_tanh(self):
        self.assertTrue(my_hard_tanh(-2.0) == -1)
        self.assertTrue(my_hard_tanh(-1.0) == -1)
        self.assertTrue(my_hard_tanh(-0.5) == -0.5)
        self.assertTrue(my_hard_tanh(0.0) == 0)
        self.assertTrue(my_hard_tanh(0.5) == 0.5)
        self.assertTrue(my_hard_tanh(1.0) == 1)
        self.assertTrue(my_hard_tanh(2.0) == 1)
        return

    def test_leaky_hard_tanh(self):
        self.assertTrue(leaky_hard_tanh(-1.0, slope=0.05) == -1.05)
        self.assertTrue(leaky_hard_tanh(0.0) == 0)
        self.assertTrue(leaky_hard_tanh(1.0, slope=0.05) == 1.05)
        return

    def test_proportional_repr(self):
        test_x = proportional_repr([1, 1])
        self.assertTrue(test_x[0] == 1 / 2)
        test_x = proportional_repr([1, 2])
        self.assertTrue(test_x[-1] == 2 / 3)
        test_x = proportional_repr([1 for i in range(10)])
        self.assertTrue(test_x[-1] == 1 / 10)
        return


if __name__ == "__main__":
    unittest.main()


# for i in range(-2, 2, 1):
#     print(i, my_hard_tanh(i))

# print(proportional_repr([1, 2]))

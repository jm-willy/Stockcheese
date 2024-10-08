import unittest

import numpy as np

from loss import MyLoss, my_loss


class TestCustomLoss(unittest.TestCase):
    def test_my_loss(self):
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = my_loss(y_true, y_pred, b=0.5)
        self.assertTrue(loss[0] == 20.666668)
        self.assertTrue(loss[1] == 34.333332)
        return

    def test_MyLoss(self):
        loss_obj = MyLoss(b=0.5)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = loss_obj(y_true, y_pred)
        self.assertTrue(loss == 27.5)
        return


if __name__ == "__main__":
    unittest.main()

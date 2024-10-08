import tensorflow as tf


@tf.function
def my_loss(y_true, y_pred, b=0.5):
    squared = tf.keras.losses.MSE(y_true, y_pred)
    absolute = tf.keras.losses.MAE(y_true, y_pred)
    x = (squared * (1 - b)) + ((absolute) * b)
    return x


class MyLoss(tf.keras.losses.Loss):
    "returns the reduced mean"

    def __init__(self, b=0.5):
        super().__init__()
        self.b = b
        return

    @tf.function
    def call(self, y_true, y_pred):
        x = my_loss(y_true, y_pred, b=self.b)
        return tf.math.reduce_mean(x, axis=-1)

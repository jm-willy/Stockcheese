import tensorflow as tf


@tf.function
def my_loss(y_true, y_pred, b=0.5):
    """
    larger b gives more weight in the average to abs over square

    y_true: target

    y_pred: output
    """
    squared = tf.keras.losses.MSE(y_true, y_pred)
    absolute = tf.keras.losses.MAE(y_true, y_pred)
    x = (squared * (1 - b)) + ((absolute) * b)
    return x


class MyLoss(tf.keras.losses.Loss):
    """
    With rduce options
    """

    def __init__(self, b=0.5, reduction="sum_over_batch_size"):
        super().__init__(reduction=reduction)
        self.b = b
        return

    @tf.function
    def call(self, y_true, y_pred):
        x = my_loss(y_true, y_pred, b=self.b)
        # x = tf.math.reduce_mean(x, axis=-1)
        return x


@tf.function
def actor_loss_func(action_probability, reward, predicted_reward):
    x = tf.keras.ops.log(action_probability) - (reward - predicted_reward)
    return x

import tensorflow as tf


@tf.function
def weighted_square_absolute(y_true, y_pred, b=0.5):
    """
    Larger b gives more weight to abs over square error

    y_true: target

    y_pred: output
    """
    squared = tf.keras.losses.MSE(y_true, y_pred)
    absolute = tf.keras.losses.MAE(y_true, y_pred)
    x = (squared * (1 - b)) + ((absolute) * b)
    return x


class WeightedSquareAbsolute(tf.keras.losses.Loss):
    """
    With reduce options

    Args:
        reduction: sum_over_batch_size, sum, None, 'none'
    """

    def __init__(self, b=0.5, reduction="sum_over_batch_size"):
        super().__init__(reduction=reduction)
        self.b = b
        return

    @tf.function
    def call(self, y_true, y_pred):
        x = weighted_square_absolute(y_true, y_pred, b=self.b)
        return x


@tf.function
def actor_loss_func(action_probability, reward, predicted_reward):
    """-log * adavantage"""
    x = -tf.keras.ops.log(action_probability) * tf.keras.ops.mean(
        reward - predicted_reward, axis=-1, keepdims=True
    )
    return x


class ActorLoss(tf.keras.losses.Loss):
    """
    With reduce options

    Args:
        reduction: sum_over_batch_size, sum, None, 'none'
    """

    def __init__(self, reduction="sum_over_batch_size"):
        super().__init__(reduction=reduction)
        return

    @tf.function
    def call(self, y_true, y_pred):
        x = actor_loss_func(y_true, y_pred[0], y_pred[1])
        return x

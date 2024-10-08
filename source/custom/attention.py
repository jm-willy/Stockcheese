import tensorflow as tf


from custom.hyperparameters import slope_init, reg
from custom.hyperparameters import heads, key_to_head_ratio


class DeepAttention(tf.keras.Model):
    """
    Deep, in contrast to wide Multi-head Attention
    """

    def __init__(self, units, depth):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=key_to_head_ratio * heads,
            attention_axes=-1,
            # output_shape=tuple((units,)),
        )
        self.dense = tf.keras.layers.Dense(units)
        self.activation = tf.keras.layers.PReLU(
            alpha_initializer=slope_init,
            activity_regularizer=reg,
        )
        return

    def call(self, inputs):
        x = self.att(inputs, inputs, inputs)
        x = self.dense(x)
        x = self.activation(x)
        return x

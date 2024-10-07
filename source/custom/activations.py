import tensorflow as tf


@tf.function
def my_hard_sigmoid(x, f_range=2):
    x = (6 * x) / f_range
    x = tf.nn.relu6(x) / 6
    return x


@tf.function
def my_hard_tanh(x, f_range=4, f_domain=2):
    x = 6 * ((x / f_range) + 0.5)
    x = tf.nn.relu6(x) / 6
    x = f_domain * (x - 0.5)
    return x


@tf.function
def leaky_hard_tanh(x, f_range=4, f_domain=2, slope=0.05):
    x = 6 * ((x / f_range) + 0.5)
    x = tf.nn.relu6(x) / 6
    x = f_domain * (x - 0.5)
    x = x + (x * slope)
    return x


class PTanh(tf.keras.layers.Layer):
    """Parametric hard tanh"""

    def __init__(
        self,
        f_range_init=tf.initializers.constant(4),
        f_domain_init=tf.initializers.constant(2),
        # slope_init=tf.initializers.constant(0.24),
    ):
        super().__init__()
        self.r_init = f_range_init
        self.d_init = f_domain_init
        # self.s_init = slope_init
        return

    def build(self, input_shape):
        self.r = self.add_weight(shape=(input_shape[-1]), initializer=self.r_init)
        self.d = self.add_weight(shape=(input_shape[-1]), initializer=self.d_init)
        return

    def call(self, inputs):
        x = my_hard_tanh(inputs, f_range=self.r, f_domain=self.d)
        return x


# def qwe(x):
#     return x


# _model_inputs = tf.keras.Input(
#     shape=(
#         2,
#         1,
#     )
# )


# x = tf.keras.layers.Dense(
#     4,
#     activation=my_hard_tanh,
#     kernel_initializer="ones",
#     bias_initializer="zeros",
# )(_model_inputs)
# x = tf.keras.layers.LSTM(
#     8,
#     activation=my_hard_tanh,
#     recurrent_activation=my_hard_sigmoid,
# )(_model_inputs)


# init_layer = tf.keras.layers.LSTM(
#     8,
#     activation=my_hard_tanh,
#     recurrent_activation=my_hard_sigmoid,
# )
# x = tf.keras.layers.Bidirectional(init_layer)(_model_inputs)

# su_model = tf.keras.Model(_model_inputs, x)
# su_model.summary(expand_nested=True, show_trainable=True)
# q = tf.constant([[1]])
# print(su_model.predict(q))

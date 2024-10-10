import tensorflow as tf


@tf.function
def my_hard_sigmoid(x):
    x = 6 * x
    x = tf.keras.ops.relu6(x) / 6
    return x


@tf.function
def my_hard_tanh(x):
    x = 3 * (x + 1)
    x = tf.keras.ops.relu6(x) / 3
    x = x - 1
    return x


@tf.function
def leaky_hard_sigmoid(x, slope=0.05):
    x = my_hard_sigmoid(x)
    x = x + (x * slope)
    return x


@tf.function
def leaky_hard_tanh(x, slope=0.05):
    x = my_hard_tanh(x)
    x = x + (x * slope)
    return x


class PTanh(tf.keras.layers.Layer):
    """Parametric hard tanh"""

    def __init__(self, slope_init=tf.initializers.constant(0.24)):
        super().__init__()
        self.slope_init = slope_init
        return

    def build(self, input_shape):
        self.slope = self.add_weight(
            shape=(input_shape[-1]), initializer=self.slope_init
        )
        return

    def call(self, inputs):
        x = my_hard_tanh(inputs, slope=self.slope)
        return x


@tf.function
def proportional_repr(x):
    x = x / tf.keras.ops.sum(x, axis=-1)
    return x


@tf.function
def madmax(x):
    """
    Proportional representation activation. Numerical stable
    Alternative to softmax.

    Squared leaky hard tanh is placed before to avoid 0 mean,
    0 division and negative probabilities.

    Hardmax name is already taken, madmax was the only
    reasonable name left.
    """
    # x = tf.keras.ops.leaky_relu(x, negative_slope=0.2)
    # x = leaky_hard_sigmoid(x)
    x = leaky_hard_tanh(x)
    # x = tf.keras.ops.absolute(x)
    x = x**2
    x = proportional_repr(x)
    return x


class Proportional(tf.keras.layers.Layer):
    """
    Proportional representation layer. Alternative to softmax.


    """

    def __init__(self):
        super().__init__()
        self.activation1 = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        return

    def call(self, inputs):
        x = self.activation1(inputs)
        x = proportional_repr(x)
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

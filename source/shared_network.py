import tensorflow as tf
from vars import vars_dict

alpha_init = vars_dict["alpha init"]
reg = vars_dict["reg"]


# (samples, time, rows, cols, channels)
shared_model_inputs = tf.keras.Input((32, 8, 8, 1))
x = shared_model_inputs

x1 = tf.keras.layers.ConvLSTM2D(
    filters=1,
    kernel_size=(7, 7),
    padding="same",
    data_format="channels_last",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    unit_forget_bias=True,
    recurrent_initializer="glorot_uniform",
    return_sequences=True,
)(x)
x2 = tf.keras.layers.ConvLSTM2D(
    filters=1,
    kernel_size=(5, 5),
    padding="same",
    data_format="channels_last",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    unit_forget_bias=True,
    recurrent_initializer="glorot_uniform",
    return_sequences=True,
)(x)
x3 = tf.keras.layers.ConvLSTM2D(
    filters=1,
    kernel_size=(3, 3),
    padding="same",
    data_format="channels_last",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    unit_forget_bias=True,
    recurrent_initializer="glorot_uniform",
    return_sequences=True,
)(x)
x = x1 + x2 + x3


x4 = tf.keras.layers.ConvLSTM2D(
    filters=1,
    kernel_size=(2, 2),
    padding="valid",
    data_format="channels_last",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    unit_forget_bias=True,
    recurrent_initializer="glorot_uniform",
    return_sequences=False,
)(x)
x4 = tf.keras.layers.Flatten()(x4)
x4 = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=64 * 4,
    attention_axes=-1,
    output_shape=tuple((512,)),
)(x4, x4, x4)


x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
    output_shape=tuple((640,)),
)(x, x, x)
x = tf.keras.layers.Concatenate(axis=-1)([x, x4])
res1 = x = tf.keras.layers.Dense(80)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
res2 = x = tf.keras.layers.Dense(80)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(80)(x + res1)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(80)(x + res2)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
    output_shape=tuple((512,)),
)(x, x, x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
shared_model = tf.keras.Model(shared_model_inputs, x)


# shared_model.summary(expand_nested=True, show_trainable=True)
# shared_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

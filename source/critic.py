import tensorflow as tf
from vars import vars_dict

alpha_init = vars_dict["alpha init"]
reg = vars_dict["reg"]


critic_input = tf.keras.Input(shape=(512,))
x = critic_input


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(32)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(4)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)


x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(1, activation="tanh")(x)
critic_model = tf.keras.Model(critic_input, x)


# critic_model.summary(expand_nested=True, show_trainable=True)
# critic_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

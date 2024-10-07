"""
actor picks a move among all possible (legal and illegal)
"""

import tensorflow as tf
from vars import vars_dict

alpha_init = vars_dict["alpha init"]
reg = vars_dict["reg"]


actor_input = tf.keras.Input((512,))
x = actor_input

x = tf.keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=64 * 2,
    attention_axes=-1,
)(x, x, x)
x = tf.keras.layers.Dense(4048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Softmax()(x)
actor_model = tf.keras.Model(actor_input, x)


# actor_model.summary(expand_nested=True, show_trainable=True)
# actor_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

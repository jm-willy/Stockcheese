import tensorflow as tf
from custom.activations import madmax
from custom.dense import DensePReLU, SixDense
from vars import sc_options, vars_dict

alpha_init = vars_dict["slope init"]
reg = vars_dict["reg"]
moves_count = vars_dict["action space size"]
input_size = sc_options["shared_model_ouput_units"]
# heads = sc_options["heads"]
# k_ratio = sc_options["keys_per_head"]

# critic_feedback_size = 2
# input_size += critic_feedback_size


actor_input = tf.keras.Input((input_size,))
x = actor_input
# x = SixDense(256)(x)
# x = DensePReLU(moves_count)(x)
# x = tf.keras.layers.Dense(moves_count)(x)
# x = tf.keras.layers.Softmax()(x)
x = tf.keras.layers.Dense(moves_count, activation=madmax)(x)
actor_model = tf.keras.Model(actor_input, x, name="ACTOR")


# actor_model.summary(expand_nested=True, show_trainable=True)
# actor_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

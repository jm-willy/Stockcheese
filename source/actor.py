import tensorflow as tf

from Stockcheese.source.custom.dense import SixDense
from Stockcheese.source.custom.dense import DensePReLU
from vars import vars_dict, sc_options


alpha_init = vars_dict["slope init"]
reg = vars_dict["reg"]
input_size = sc_options["shared_model_ouput_units"]
heads = sc_options["heads"]
k_ratio = sc_options["keys_per_head"]
moves_count = sc_options["action space size"]

critic_feedback_size = 1
input_size += critic_feedback_size


actor_input = tf.keras.Input((input_size,))
x = actor_input
# x = SixDense(256)(x)
x = DensePReLU(moves_count)(x)
x = tf.keras.layers.Softmax()(x)
actor_model = tf.keras.Model(actor_input, x, name="ACTOR")


# actor_model.summary(expand_nested=True, show_trainable=True)
# actor_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

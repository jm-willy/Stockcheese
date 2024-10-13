import tensorflow as tf
from custom.dense import DensePReLU, SixDense
from vars import sc_options, vars_dict

alpha_init = vars_dict["slope init"]
reg = vars_dict["reg"]
input_size = sc_options["shared_model_ouput_units"]
heads = sc_options["heads"]
k_ratio = sc_options["keys_per_head"]


critic_input = tf.keras.Input(shape=(input_size,))
x = critic_input
# x = SixDense(128)(x)
# x = SixDense(64)(x)
# x = SixDense(32)(x)
# x = SixDense(16)(x)
x = DensePReLU(8)(x)
x = DensePReLU(4)(x)
x = DensePReLU(2)(x)
x = tf.keras.layers.Dense(1, activation="tanh")(x)
critic_model = tf.keras.Model(critic_input, x, name="CRITIC")


# critic_model.summary(expand_nested=True, show_trainable=True)
# critic_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

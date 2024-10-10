import tensorflow as tf
from custom.convolutional import Conv3dBlock
from custom.dense import DensePReLU
from vars import vars_dict, sc_options

from custom.dense import SixDense


alpha_init = vars_dict["slope init"]
reg = vars_dict["reg"]
output_units = sc_options["shared_model_ouput_units"]
rem = sc_options["remember"]
heads = sc_options["heads"]
k_ratio = sc_options["keys_per_head"]


# (None, time, rows, cols, channels)
shared_model_inputs = tf.keras.Input((rem, 8, 8, 1))
x = shared_model_inputs
x = Conv3dBlock()(x)
x = Conv3dBlock()(x)
x = Conv3dBlock()(x)
# x = tf.keras.layers.Concatenate(axis=-1)(
#     [
#         tf.keras.layers.Flatten()(x),
#         tf.keras.layers.Flatten()(shared_model_inputs),
#     ]
# )
x = tf.keras.layers.Flatten()(x)  ##########3
x = DensePReLU(640)(x)


# res1 = x = SixDense(80)(x)
# res2 = x = SixDense(80)(x)
# res3 = x = SixDense(80)(x)

# res1 = x = SixDense(80)(x + res1)
# res2 = x = SixDense(80)(x + res2)
# res3 = x = SixDense(80)(x + res3)

# res1 = x = SixDense(80)(x + res1)
# res2 = x = SixDense(80)(x + res2)
# res3 = x = SixDense(80)(x + res3)


# x = SixDense(output_units)(x)
x = DensePReLU(output_units)(x)
shared_model = tf.keras.Model(shared_model_inputs, x, name="SHARED.MODEL")


# shared_model.summary(expand_nested=True, show_trainable=True)
# shared_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# print("compiled")

import tensorflow as tf
from actor import actor_model
from critic import critic_model
from shared_network import shared_model
from time_utils import date_time_print
from vars import sc_options

input_size = sc_options["remember"]


full_input = tf.keras.layers.Input(shape=(input_size, 8, 8, 1))
x = shared_model(full_input)
critic_feedback = critic_model(x)
# x = tf.keras.layers.Concatenate(axis=-1)([critic_feedback, x, critic_feedback])
action = actor_model(x)
model = tf.keras.Model(
    inputs=full_input, outputs=[critic_feedback, action], name="FULL_MODEL"
)

# print()
# date_time_print("shared_model summary:")
# shared_model.summary(expand_nested=True, show_trainable=True)
# print()
# date_time_print("critic_model summary:")
# critic_model.summary(expand_nested=True, show_trainable=True)
# print()
# date_time_print("actor_model summary:")
# actor_model.summary(expand_nested=True, show_trainable=True)

# print()
# date_time_print("model summary:")
# model.summary(expand_nested=False, show_trainable=True)

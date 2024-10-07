import tensorflow as tf

from actor import actor_model
from critic import critic_model
from shared_network import shared_model
from time_utils import date_time_print


inputs = tf.keras.layers.Input(shape=(32, 8, 8, 1))
x = shared_model(inputs)
critic_feedback = critic_model(x)
action = actor_model(x * critic_feedback)
model = tf.keras.Model(inputs=inputs, outputs=[critic_feedback, action])

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

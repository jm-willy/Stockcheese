import tensorflow as tf

from actor import actor_model
from critic import critic_model
from shared_network import shared_model
from time_utils import date_time_print

inputs = tf.keras.layers.Input(shape=(32, 64))
x = shared_model(inputs)
critic_feedback = critic_model(x)  # gradient stops below here
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.LayerNormalization()(x)
# y = tf.repeat(critic_feedback, repeats=[32], axis=-1)
#
# # print(y)
# # x = tf.expand_dims(x, axis=0)
# x = tf.keras.layers.concatenate([x, y], axis=-1)
# for i in range(1):
#     x = tf.keras.layers.concatenate([x, y], axis=-1)
#     # x = tf.Variable(tf.concat([x, y], axis=-1))
action = actor_model(x + critic_feedback)  # to here
model = tf.keras.Model(inputs=inputs, outputs=[critic_feedback, action])

# shared_model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
# print('compiled')

date_time_print('shared_model summary:')
shared_model.summary()

date_time_print('critic_model summary:')
critic_model.summary()

date_time_print('actor_model summary:')
actor_model.summary()

date_time_print('model summary:')
model.summary()

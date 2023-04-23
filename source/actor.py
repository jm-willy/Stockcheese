import tensorflow as tf

from sub_model import sub_model
from vars import vars_dict

alpha_init = vars_dict['alpha init']
reg = vars_dict['reg']

actor_input = tf.keras.Input((2048,))

x = tf.keras.layers.Dense(2048)(actor_input)
res = x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x + res)

x = tf.keras.layers.Dense(4048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.LayerNormalization()(x)
actor_output = tf.keras.layers.Softmax()(x)
actor_model = tf.keras.Model(actor_input, actor_output)

# actor_model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
# print('compiled')

import tensorflow as tf

from vars import vars_dict

alpha_init = vars_dict['alpha init']
reg = vars_dict['reg']

sub_model_inputs = tf.keras.Input(shape=(2048,))
x = tf.keras.layers.Dense(2048)(sub_model_inputs)
res = x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)

x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x + res)

sub_model_outputs = tf.keras.layers.LayerNormalization()(x)
sub_model = tf.keras.Model(sub_model_inputs, sub_model_outputs)

import tensorflow as tf

from vars import vars_dict

alpha_init = vars_dict['alpha init']
reg = vars_dict['reg']

critic_input = tf.keras.Input((32, 64))

x = tf.keras.layers.Flatten()(critic_input)
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.reshape(x, shape=(tf.shape(x)[0], 16, 32))
x = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.reshape(x, shape=(tf.shape(x)[0], 8, 16,))
x = tf.keras.layers.LSTM(4, activation='tanh', return_sequences=True)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.LayerNormalization()(x)

x = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.Dense(16)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.LayerNormalization()(x)

x = tf.keras.layers.Dense(4)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.Dense(2)(x)
x = tf.keras.layers.PReLU(alpha_initializer=alpha_init, activity_regularizer=reg)(x)
x = tf.keras.layers.LayerNormalization()(x)

critic_output = tf.keras.layers.Dense(1, activation='tanh')(x)
critic_model = tf.keras.Model(critic_input, critic_output)

# critic_model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
# print('compiled')

import tensorflow as tf

from sub_model import sub_model

shared_model_inputs = tf.keras.Input((32, 64))
x = tf.keras.layers.Dropout(4 / 64)(shared_model_inputs)

lstm_res = x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)
x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.LayerNormalization()(x)
dense_res = x = sub_model(x)
x = tf.reshape(x, shape=(tf.shape(x)[0], 32, 64))

for i in range(0, 2):
    lstm_res = x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(x + lstm_res)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    dense_res = x = sub_model(x + dense_res)
    x = tf.reshape(x, shape=(tf.shape(x)[0], 32, 64))

x = tf.keras.layers.LSTM(64, return_sequences=True)(x + lstm_res)
shared_model_outputs = tf.keras.layers.Dropout(4 / 64)(x)
shared_model = tf.keras.Model(shared_model_inputs, shared_model_outputs)

# shared_model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
# print('compiled')

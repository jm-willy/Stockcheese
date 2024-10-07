import tensorflow as tf


slope_init = tf.initializers.constant(0.24)
reg = tf.keras.regularizers.L2(l2=0.00)
heads = 2
key_to_head_ratio = 48

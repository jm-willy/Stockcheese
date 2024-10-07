import tensorflow as tf


class Conv3dBlock(tf.keras.Model):
    """Limited use, add args"""

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=(3, 3, 6),
            strides=(1, 1, 1),
            padding="same",
            data_format="channels_last",
            dilation_rate=(1, 1, 1),
            groups=1,
            activation="leaky_relu",
        )
        self.conv2 = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=(6, 6, 3),
            strides=(1, 1, 1),
            padding="same",
            data_format="channels_last",
            dilation_rate=(1, 1, 1),
            groups=1,
            activation="leaky_relu",
        )
        self.conv3 = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=(5, 5, 5),
            strides=(1, 1, 1),
            padding="same",
            data_format="channels_last",
            dilation_rate=(1, 1, 1),
            groups=1,
            activation="leaky_relu",
        )
        return

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs) + self.conv2(inputs) + self.conv3(inputs)
        return x

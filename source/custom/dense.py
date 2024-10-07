import tensorflow as tf


from hyperparameters import slope_init, reg


class DensePReLU(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.L = tf.keras.layers.Dense(units)
        self.A = tf.keras.layers.PReLU(
            slope_initializer=slope_init,
            activity_regularizer=reg,
        )
        return

    def call(self, inputs):
        x = self.L(inputs)
        x = self.A(x)
        return x


class SixDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.l1 = DensePReLU(units)
        self.l2 = DensePReLU(units)
        self.l3 = DensePReLU(units)
        self.l4 = DensePReLU(units)
        self.l5 = DensePReLU(units)
        self.l6 = DensePReLU(units)
        return

    def call(self, inputs):
        r1 = x = self.l1(inputs)
        r2 = x = self.l2(x)
        r3 = x = self.l3(x)

        x = self.l4(x + r1)
        x = self.l5(x + r2)
        x = self.l6(x + r3)
        return x

import tensorflow as tf


class MultiplicativeUnit:
    def __init__(self, kernel_size=3, dilation_rate=1, filters=64):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.filters = filters

    def __call__(self, h):
        with tf.name_scope('MU'):
            g1 = tf.layers.conv2d(
                h,
                self.filters,
                self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            g2 = tf.layers.conv2d(
                h,
                self.filters,
                self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            g3 = tf.layers.conv2d(
                h,
                self.filters,
                self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            u = tf.layers.conv2d(
                h,
                self.filters,
                self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding='same',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu


class ResidualMultiplicativeBlock:
    def __init__(self, kernel_size=3, dilation_rate=1, filters=64):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.filters = filters

    def __call__(self, h):
        with tf.name_scope('RMU'):
            h1 = tf.layers.conv2d(
                h,
                self.filters / 2,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
            )

            h2 = MultiplicativeUnit(kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, filters=self.filters / 2)(h1)

            h3 = MultiplicativeUnit(kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, filters=self.filters / 2)(h2)

            h4 = tf.layers.conv2d(
                h3,
                self.filters,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            rmb = tf.add(h, h4)
            return rmb


class CascadeMultiplicativeUnit:
    def __init__(self, kernel_size=3, dilation_rate=1, filters=64):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.filters = filters

    def __call__(self, prev, current):
        with tf.name_scope('CMU'):
            mu_template = tf.make_template('MU_couple', MultiplicativeUnit(kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, filters=self.filters), create_scope_now_=True)
            CurrMU = MultiplicativeUnit(kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, filters=self.filters)

            h1 = mu_template(mu_template(prev))
            h2 = CurrMU(current)

            h = h1 + h2

            o = tf.layers.conv2d(
                h,
                self.filters,
                self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            cmu = tf.multiply(o, tf.tanh(h))
            return cmu


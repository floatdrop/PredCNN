import tensorflow as tf
from units import ResidualMultiplicativeBlock as RMB, CascadeMultiplicativeUnit as CMU


class VideoPixelNetworkModel:
    def __init__(self, config):
        self.config = config

        with tf.name_scope('inputs'):
            self.sequences = tf.placeholder(tf.float32,
                                            shape=[None, config.truncated_steps + 1] + config.input_shape,
                                            name='sequences')

        self.build_model()

    def encoder(self, h):
        for i in range(self.config.encoder_rmb_num):
            if self.config.encoder_rmb_dilation:
                h = RMB(dilation_rate=self.config.encoder_rmb_dilation_scheme[i], filters=self.config.rmb_c)(h)
            else:
                h = RMB(filters=self.config.rmb_c)(h)

        return h

    def decoder(self, h):
        with tf.variable_scope('decoders'):
            for i in range(0, self.config.decoder_rmb_num):
                h = RMB(filters=self.config.rmb_c)(h)

            h = tf.layers.conv2d(
                h,
                256,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            return h

    def build_model(self):
        with tf.name_scope('training_graph'):
            encoders = []
            for i in range(self.config.truncated_steps):
                with tf.name_scope('encoder_%i' % i):
                    encoders.append(self.encoder(self.sequences[:, i]))

            while len(encoders) > 1:
                new_layer = []
                for first, second in zip(encoders, encoders[1:]):
                    with tf.name_scope('CMU_%i_%i' % (len(encoders) - 1, i)):
                        new_layer.append(CMU(filters=self.config.rmb_c)(first, second))
                encoders = new_layer

            self.output = self.decoder(encoders[0])

        with tf.name_scope('loss'):
            labels = tf.one_hot(tf.cast(tf.squeeze(self.sequences[:, -1]), tf.int32),
                                256,
                                axis=-1,
                                dtype=tf.float32)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=labels))

            # self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, self.output))))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        self.summaries = tf.summary.merge_all('vpn')

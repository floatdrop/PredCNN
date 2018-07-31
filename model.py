import tensorflow as tf
from units import ResidualMultiplicativeBlock as RMB, CascadeMultiplicativeUnit as CMU


class PredCNN:
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
        with tf.variable_scope('decoder'):
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
            encoder_template = tf.make_template('encoder', self.encoder, create_scope_now_=True)

            encoders = []

            for i in range(self.config.truncated_steps):
                encoders.append(encoder_template(self.sequences[:, i]))

            l = 0
            while len(encoders) > 1:
                new_layer = []
                cmu_template = tf.make_template('CMU_layer_%i' % l, CMU(filters=self.config.rmb_c), create_scope_now_=True)
                for first, second in zip(encoders, encoders[1:]):
                    new_layer.append(cmu_template(first, second))
                encoders = new_layer
                l += 1

            self.output = self.decoder(encoders[0])

        with tf.name_scope('loss'):
            labels = tf.one_hot(tf.cast(tf.squeeze(self.sequences[:, -1]), tf.int32),
                                256,
                                axis=-1,
                                dtype=tf.float32)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=labels))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        self.summaries = tf.summary.merge_all('vpn')

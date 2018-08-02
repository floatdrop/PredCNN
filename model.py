import tensorflow as tf
from units import ResidualMultiplicativeBlock as RMB, CascadeMultiplicativeUnit as CMU, MultiplicativeUnit


class PredCNN:
    def __init__(self, config):
        self.config = config

        with tf.name_scope('inputs'):
            self.sequences = tf.placeholder(tf.float32,
                                            shape=[None, config.truncated_steps] + config.input_shape,
                                            name='sequences')
            self.targets = tf.placeholder(tf.float32,
                                            shape=[None, 1] + config.input_shape,
                                            name='targets')

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

            with tf.name_scope('Last_RMU'):
                h1 = tf.layers.conv2d(
                    h,
                    self.config.rmb_c / 2,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )

                h2 = MultiplicativeUnit(filters=self.config.rmb_c / 2)(h1)

                h3 = MultiplicativeUnit(filters=self.config.rmb_c / 2)(h2)

                h4 = tf.layers.conv2d(
                    h3,
                    256,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )

                return h4

    def build_model(self):

        with tf.name_scope('training_graph'):
            encoder_template = tf.make_template('encoder', self.encoder, create_scope_now_=True)

            encoders = []

            for i in range(self.config.truncated_steps):
                encoders.append(encoder_template(self.sequences[:, i]))

            l = 0
            while len(encoders) > 1:
                new_layer = []
                # cmu_template = tf.make_template('CMU_layer_%i' % l, CMU(filters=self.config.rmb_c), create_scope_now_=True)
                for first, second in zip(encoders, encoders[1:]):
                    new_layer.append(CMU(filters=self.config.rmb_c)(first, second))
                encoders = new_layer
                l += 1

            self.output = self.decoder(encoders[0])

        with tf.name_scope('loss'):
            labels = tf.one_hot(tf.cast(tf.squeeze(self.targets), tf.int32),
                                256,
                                axis=-1,
                                dtype=tf.float32)

            # output = numpy.argmax(self.output, axis=3)[0]
            # self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.targets, self.output))))
            # self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.targets, self.output), 2))
            
            self.loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=self.output, targets=labels, pos_weight=0.05))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        self.summaries = tf.summary.merge_all('predcnn')

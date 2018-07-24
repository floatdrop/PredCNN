import tensorflow as tf
from logger import Logger


class VideoPixelNetworkModel:
    def __init__(self, config):
        self.config = config

        with tf.name_scope('inputs'):
            self.sequences = tf.placeholder(tf.float32,
                                            shape=[None, config.truncated_steps + 1] + config.input_shape,
                                            name='sequences')

        self.build_model()

    def multiplicative_unit_without_mask(self, h, dilation_rate, scope):
        with tf.variable_scope('multiplicative_unit_without_mask_' + scope):
            g1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g1'
            )

            g2 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g2'
            )

            g3 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g3'
            )

            u = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='u'
            )

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu

    def residual_multiplicative_block_without_mask(self, h, dilation_rate, scope):
        with tf.variable_scope('residual_multiplicative_block_without_mask_' + scope):
            h1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h1'
            )

            h2 = self.multiplicative_unit_without_mask(h1, dilation_rate, '1')

            h3 = self.multiplicative_unit_without_mask(h2, dilation_rate, '2')

            h4 = tf.layers.conv2d(
                h3,
                2 * self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h4'
            )

            rmb = tf.add(h, h4)

            return rmb


    def cascade_multiplicative_unit(self, prev_h, curr_h, dilation_rate, scope):
        with tf.variable_scope('cascade_multiplicative_unit_' + scope, reuse=tf.AUTO_REUSE):
            h1 = self.multiplicative_unit_without_mask(prev_h, dilation_rate, 'prev_mu_1')
            h1 = self.multiplicative_unit_without_mask(h1, dilation_rate, 'prev_mu_1')

            h2 = self.multiplicative_unit_without_mask(curr_h, dilation_rate, 'curr_mu_1')

            h = h1 + h2

            o = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='o'
            )

            cmu = tf.multiply(o, tf.tanh(h))

            return cmu

    def resolution_preserving_cnn_encoders(self, x):
        with tf.variable_scope('resolution_preserving_cnn_encoders'):
            x = tf.layers.conv2d(
                x,
                2 * self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='input_conv'
            )
            for i in range(self.config.encoder_rmb_num):
                if self.config.encoder_rmb_dilation:
                    x = self.residual_multiplicative_block_without_mask(x, self.config.encoder_rmb_dilation_scheme[i],
                                                                        str(i))
                else:
                    x = self.residual_multiplicative_block_without_mask(x, 1, str(i))

            return x

    def resolution_preserving_cnn_decoders(self, h, x):
        with tf.variable_scope('resolution_preserving_cnn_decoders'):
            h = self.residual_multiplicative_block_without_mask(h, 1, '00')

            h = (x, h)
            h = self.residual_multiplicative_block_without_mask(h, 1, str(1))

            for i in range(2, self.config.decoder_rmb_num):
                h = self.residual_multiplicative_block_with_mask(h, 1, str(i))

            return h

    def cmu_template(self, prev_x, curr_x):
        encoder_h = self.cascade_multiplicative_unit(prev_x, curr_x, 1, 'cmu')
        return encoder_h

    def build_model(self):
        encoder_network_template = tf.make_template('vpn_encoder', self.resolution_preserving_cnn_encoders)
        cmu_network_template = tf.make_template('vpn_cmu', self.cmu_template)
        decoder_network_template = tf.make_template('vpn_decoder', self.resolution_preserving_cnn_decoders)

        with tf.name_scope('training_graph'):
            encoders = [encoder_network_template(self.sequences[:, i]) for i in range(self.config.truncated_steps)]

            while len(encoders) > 1:
                encoders = [cmu_network_template(first, second) for first, second in zip(encoders, encoders[1:])]

            self.output = decoder_network_template(encoders[0])

        with tf.name_scope('loss'):
            labels = tf.one_hot(tf.cast(tf.squeeze(self.sequences[:, 1:]), tf.int32),
                                256,
                                axis=-1,
                                dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=labels))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('test_frames'):
            self.test_summaries = []
            for i in range(self.config.truncated_steps):
                Logger.summarize_images(tf.expand_dims(tf.cast(tf.arg_max(self.inference_output, 3), tf.float32), 3),
                                        'test_frame_{0}'.format(i), 'vpn_test_{0}'.format(i), 1)
                self.test_summaries.append(tf.summary.merge_all('vpn_test_{0}'.format(i)))

        self.summaries = tf.summary.merge_all('vpn')

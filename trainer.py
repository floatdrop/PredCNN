import tensorflow as tf
import numpy as np
from logger import Logger
import os
from imageio import imwrite
from tqdm import tqdm
import imageio


class Trainer:
    def __init__(self, sess, model, data_generator, config):
        self.sess = sess
        self.model = model
        self.data_generator = data_generator
        self.config = config

        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None

        # init the global step , the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        if not os.path.exists(self.config.summary_dir):
            os.makedirs(self.config.summary_dir)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        self.logger = Logger(self.sess, self.config.summary_dir)

        if self.config.load:
            self.load()

    def save(self):
        self.saver.save(self.sess, self.config.checkpoint_dir, self.global_step_tensor)
        Logger.info("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            Logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            Logger.info("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def train(self):
        Logger.info("Starting training...")

        for epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.epochs_num):
            losses = []

            epoch = self.cur_epoch_tensor.eval(self.sess)

            for itr in tqdm(range(self.config.iters_per_epoch)):
                train_batch = self.data_generator.next_batch()

                feed_dict = {
                    self.model.sequences: train_batch[:, :-1],
                    self.model.targets: train_batch[:, -1:]
                }
                loss, _ = self.sess.run([self.model.loss, self.model.optimizer], feed_dict)
                losses.append(loss)

                self.sess.run(self.global_step_assign_op,
                              {self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

            Logger.info('epoch #{0}:    loss={1}'.format(epoch, np.mean(losses)))
            self.logger.add_scalar_summary(self.global_step_tensor.eval(self.sess), {'train_loss': np.mean(losses)})
            self.sess.run(self.cur_epoch_assign_op, {self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            if epoch > 0 and epoch % self.config.test_every == 0:
                self.test(epoch)
                self.save()

        Logger.info("Training finished")

    def test(self, step):
        Logger.info("Starting testing...")
        p =  os.path.join(self.config.summary_dir,'test', str(step))
        if not os.path.exists(p):
            os.makedirs(p)
        for i in range(5):
            test_batch = self.data_generator.test_batch()

            imageio.mimwrite(os.path.join(p, '%i_gt.gif' % i), test_batch[0].astype(np.uint8))

            images = np.array(test_batch[:, :self.config.truncated_steps])
            for j in range(20 - self.config.truncated_steps):
                feed_dict = {self.model.sequences: images[:, j:j+self.config.truncated_steps]}
                output = self.sess.run(self.model.output, feed_dict)
                output = np.expand_dims(np.argmax(output, axis=3)[0], axis=2)
                images[0, images.shape[1] - 1, :, :, :] = output
                images = np.append(images, [[output]], axis=1)

            imageio.mimwrite(os.path.join(p, '%i_pd.gif' % i), images[0].astype(np.uint8))
        
        Logger.info("Testing finished")

from os import path
import tensorflow as tf
import config
from model import PredCNN
from data_generator import GenerateData
from trainer import Trainer
from logger import Logger

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', "predcnn", """ experiment name """)
tf.app.flags.DEFINE_boolean('train', True, """ train flag """)
tf.app.flags.DEFINE_boolean('overfitting', False, """ overfitting flag """)
tf.app.flags.DEFINE_boolean('load', True, """ model loading flag """)
tf.app.flags.DEFINE_integer('batch_size', 1, """ batch size for training """)
tf.app.flags.DEFINE_string('data_dir', "data.dat", """ data directory """)
tf.app.flags.DEFINE_string('exp_dir', "/tmp/predcnn", """ experiment directory """)


def main(_):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    if FLAGS.train is not None:
        config.train = FLAGS.train
    if FLAGS.load is not None:
        config.load = FLAGS.load
    if FLAGS.batch_size is not None:
        config.batch_size = FLAGS.batch_size
    if FLAGS.overfitting is not None:
        config.overfitting = FLAGS.overfitting
        if FLAGS.overfitting:
            config.batch_size = 1
            config.train_sequences_num = 1
            config.iters_per_epoch = 1
    if FLAGS.data_dir is not None:
        config.data_dir = FLAGS.data_dir
    if FLAGS.exp_dir is not None:
        config.summary_dir = path.join(FLAGS.exp_dir, FLAGS.name)
        config.checkpoint_dir = path.join(config.summary_dir, 'checkpoints')

    Logger.info('Starting building the model...')
    vpn = PredCNN(config)
    data_generator = GenerateData(config)
    trainer = Trainer(sess, vpn, data_generator, config)
    Logger.info('Finished building the model')

    if config.train:
        trainer.train()
    trainer.test()


if __name__ == '__main__':
    tf.app.run()

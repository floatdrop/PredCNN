import tensorflow as tf
from units import *


tf.reset_default_graph()
with tf.Graph().as_default() as g:
    x = tf.random_normal((1, 300, 300, 32)) # (batch_size, width, height, channel)
    with tf.variable_scope("tcn"):
        conv = CascadeMultiplicativeUnit(kernel_size=3, filters=32, dilation_rate=1)
    output = conv(x, x)
    init = tf.global_variables_initializer()
    
with tf.Session(graph=g) as sess:
    # Run the initializer
    sess.run(init)
    res = sess.run(output)
    # print(inputs.shape)
    # print(inputs[0, :, 0])
    print(res.shape)    
    print(res[0, :, 0])
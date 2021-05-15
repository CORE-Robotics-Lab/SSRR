import tensorflow as tf
import numpy as np

REG_VARS = 'reg_vars'


def linear(X, dout, name, bias=True):
    with tf.variable_scope(name):
        dX = int(X.get_shape()[-1])
        W = tf.get_variable('W', shape=(dX, dout))
        tf.add_to_collection(REG_VARS, W)
        if bias:
            b = tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32)))
        else:
            b = 0
    return tf.matmul(X, W) + b


def relu_layer(X, dout, name):
    return tf.nn.relu(linear(X, dout, name))


def get_session_config():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    return session_config

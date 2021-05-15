import tensorflow as tf
from inverse_rl.models.tf_util import relu_layer, linear


def relu_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    out = linear(out, dout=dout, name='lfinal')
    return out

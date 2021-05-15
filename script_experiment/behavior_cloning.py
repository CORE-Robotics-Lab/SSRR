# From D-REX code https://github.com/dsbrown1331/CoRL2019-DREX

import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import gym
from script_experiment.tf_commons.ops import Linear
from global_utils.utils import set_seed


class Policy(object):
    def __init__(self, env, num_layers=4, embed_size=256):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            self.inp = tf.placeholder(tf.float32, [None, ob_dim])
            self.l = tf.placeholder(tf.float32, [None, ac_dim])
            self.l2_reg = tf.placeholder(tf.float32, [])

            with tf.variable_scope('weights') as param_scope:
                self.param_scope = param_scope

                fcs = []
                last_dims = ob_dim
                for l in range(num_layers):
                    fcs.append(Linear('fc%d' % (l + 1), last_dims,
                                      embed_size))  # (l+1) is gross, but for backward compatibility
                    last_dims = embed_size
                fcs.append(Linear('fc%d' % (num_layers + 1), last_dims, ac_dim))

            # build graph
            def _build(x):
                for fc in fcs[:-1]:
                    x = tf.nn.relu(fc(x))
                pred_a = fcs[-1](x)
                return pred_a

            self.ac = _build(self.inp)

            loss = tf.reduce_sum((self.ac - self.l) ** 2, axis=1)
            self.loss = tf.reduce_mean(loss, axis=0)

            weight_decay = 0.
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.w ** 2)

            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss + self.l2_loss, var_list=self.parameters(train=True))

            self.saver = tf.train.Saver(var_list=self.parameters(train=False), max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self, train=False):
        with self.graph.as_default():
            if train:
                return tf.trainable_variables(self.param_scope.name)
            else:
                return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.param_scope.name)

    def train(self, D, batch_size, iter, l2_reg, debug=False):
        sess = self.sess

        obs, acs, _ = D

        idxes = np.random.permutation(len(obs) - 1)
        train_idxes = idxes[:int(len(obs) * 0.8)]
        valid_idxes = idxes[int(len(obs) * 0.8):]

        def _batch(idx_list):
            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list, batch_size, replace=False)
            else:
                idxes = idx_list

            batch = []
            for i in idxes:
                batch.append((obs[i], acs[i]))
            b_s, b_a = [np.array(e) for e in zip(*batch)]

            return b_s, b_a

        for it in tqdm(range(iter), dynamic_ncols=True):
            b_s, b_a = _batch(train_idxes)

            with self.graph.as_default():
                loss, l2_loss, _ = sess.run([self.loss, self.l2_loss, self.update_op], feed_dict={
                    self.inp: b_s,
                    self.l: b_a,
                    self.l2_reg: l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_s, b_a = _batch(valid_idxes)
                    valid_loss = sess.run(self.loss, feed_dict={
                        self.inp: b_s,
                        self.l: b_a,
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f' % (loss, l2_loss, valid_loss)))

    def act(self, observation, reward, done, clip=False):
        sess = self.sess

        with self.graph.as_default():
            ac = sess.run(self.ac, feed_dict={self.inp: observation[None]})[0]

        if clip:
            ac = np.clip(ac, self.action_space.low, self.action_space.high)

        return ac

    def save(self, path):
        with self.graph.as_default():
            self.saver.save(self.sess, path, write_meta_graph=False)

    def load(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

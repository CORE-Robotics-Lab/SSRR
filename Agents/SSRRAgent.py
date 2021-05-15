import numpy as np
import tensorflow as tf
from tqdm import tqdm


class SSRRAgent:
    def __init__(self, include_action, ob_dim, ac_dim, layers, batch_size):
        self.include_action = include_action
        self.input_dims = ob_dim + ac_dim if include_action else ob_dim
        self.batch_size = batch_size

        self._build_graph(layers)

    def _build_graph(self, layers):
        with tf.variable_scope('weights') as param_scope:
            self.input_ph = tf.placeholder(tf.float32, [None, self.input_dims])
            self.snippet_split = tf.placeholder(tf.int32, [self.batch_size])
            self.label = tf.placeholder(tf.float32, [self.batch_size])
            self.l2_reg = tf.placeholder(tf.float32, [])

            self.param_scope = param_scope

            fcs = []
            output = self.input_ph
            for layer in layers:
                fcs.append(tf.layers.Dense(layer, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.001)))
                output = fcs[-1](output)
            fcs.append(tf.layers.Dense(1, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.001)))
            output = fcs[-1](output)
            self.reward_output = tf.squeeze(output, axis=1)

            self.snippet_value = tf.stack([tf.reduce_sum(rs_x) for rs_x in tf.split(self.reward_output, self.snippet_split, axis=0)], axis=0)
            self.loss = tf.reduce_mean(tf.square(self.snippet_value - self.label), axis=0)

            weight_decay = 0.0
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.kernel ** 2)
            self.l2_loss = self.l2_reg * weight_decay

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss + self.l2_loss, var_list=self.parameters(train=True))
        self.saver = tf.train.Saver(var_list=self.parameters(train=False), max_to_keep=0)

    def parameters(self, train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.param_scope.name)

    def input_preprocess(self, obs, acs):
        assert len(obs) == len(acs) or len(obs) == len(acs) + 1
        return np.concatenate((obs[:len(acs)], acs), axis=1) if self.include_action else obs

    def train(self, D, iter=10000, l2_reg=0.01, debug=False, early_term=False):
        """
        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
                while
                    sigma^{1,2}: shape of [steps,in_dims]
                    mu : 0 or 1
            l2_reg
            debug: print training statistics
            early_term:  training will be early-terminate when validation accuracy is larger than 95%
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D) * 0.8)]
        valid_idxes = idxes[int(len(D) * 0.8):]

        def _load(idxes):
            batch = []

            for i in idxes:
                batch.append(D[i])

            b_x, b_l = zip(*batch)
            x_split = np.array([len(x) for x in b_x])
            b_x, b_l = np.concatenate(b_x, axis=0), np.array(b_l)

            return b_x.astype(np.float32), x_split, b_l

        def _batch(idx_list):
            if len(idx_list) > self.batch_size:
                idxes = np.random.choice(idx_list, self.batch_size, replace=False)
            else:
                idxes = idx_list

            return _load(idxes)

        for it in tqdm(range(iter), dynamic_ncols=True):
            b_x, x_split, b_l = _batch(train_idxes)

            loss, l2_loss, _ = sess.run([self.loss, self.l2_loss, self.update_op], feed_dict={
                self.input_ph: b_x,
                self.snippet_split: x_split,
                self.label: b_l,
                self.l2_reg: l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_x, x_split, b_l = _batch(valid_idxes)
                    valid_loss = sess.run(self.loss, feed_dict={
                        self.input_ph: b_x,
                        self.snippet_split: x_split,
                        self.label: b_l
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f' % (loss, l2_loss, valid_loss)))

            if early_term and loss <= 1e-3:
                print('loss: %f (l2_loss: %f)' % (loss, l2_loss))
                print('early termination@%08d' % it)
                break

    def get_reward(self, obs, acs, batch_size=1024):
        sess = tf.get_default_session()

        inp = self.input_preprocess(obs, acs)

        b_r = []
        for i in range(0, len(obs), batch_size):
            r = sess.run(self.reward_output, feed_dict={
                self.input_ph: inp[i:i + batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r, axis=0)

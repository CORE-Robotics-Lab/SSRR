# adjusted from rl-baselines-zoo:  https://github.com/araffin/rl-baselines-zoo

import os
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from argparse import Namespace

from Agents.SSRRAgent import SSRRAgent


class CustomNormalizedReward(gym.Wrapper):
    def __init__(self, env, model_dir, ctrl_coeff, alive_bonus):
        super(CustomNormalizedReward, self).__init__(env)

        ob_shape = env.observation_space.shape
        ac_dims = env.action_space.shape[-1]

        self.ctrl_coeff = ctrl_coeff
        self.alive_bonus = alive_bonus

        self.graph = tf.Graph()
        config = tf.ConfigProto(device_count={'GPU': 0})  # Run on CPU
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                print(os.path.realpath(model_dir))
                with open(str(Path(model_dir) / 'args.txt')) as f:
                    args = eval(f.read())

                models = []
                for i in range(args.num_models):
                    with tf.variable_scope('model_%d' % i):
                        model = SSRRAgent(include_action=args.include_action,
                                              ob_dim=ob_shape[-1],
                                              ac_dim=ac_dims,
                                              layers=[256, 256],
                                              batch_size=64)
                        model.saver.restore(self.sess, os.path.join(model_dir, 'model_%d.ckpt' % i))
                        models.append(model)
                self.models = models
        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(self.models))]
        self.cliprew = 10.
        self.epsilon = 1e-8

    def step(self, action):
        assert len(action.shape) == 1
        ob, reward, done, info = self.env.step(action)
        r_hats = 0
        with self.graph.as_default():
            with self.sess.as_default():
                for model, rms in zip(self.models, self.rew_rms):
                    # Preference based reward
                    r_hat = model.get_reward([ob], [action])[0]

                    # Normalize
                    # rms.update(np.array([r_hat]))
                    # r_hat = np.clip(r_hat / np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)

                    # Sum-up each models' reward
                    r_hats += r_hat
        rews = r_hats / len(self.models)  # - self.ctrl_coeff*np.sum(action**2)
        # rews += self.alive_bonus

        return ob, rews, done, info


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

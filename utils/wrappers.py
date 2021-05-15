import gym
from gym.wrappers import TimeLimit
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path
from argparse import Namespace


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


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
                sys.path.append("/home/zac/Programming/Zac-Meta-D-REX")
                # dir_path = os.path.dirname(os.path.realpath(__file__))
                # sys.path.append(os.path.join(dir_path, '..', '..', '..', '..'))
                from SSRRAgent import SSRRAgent

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

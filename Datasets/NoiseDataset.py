import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class NoiseDataset(object):
    def __init__(self, env):
        self.env = env
        self.trajs = None

    def load_prebuilt(self, fname):
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.trajs = pickle.load(f)
            return True
        else:
            return False

    def sample_not_paired(self, num_samples, min_steps, max_steps, sigmoid_params, include_action=False):
        def sigmoid(p, x):
            x0, y0, c, k = p
            y = c / (1 + np.exp(-k * (x - x0))) + y0
            return y

        D = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:

            for _ in tqdm(range(num_samples), desc="sampling"):
                x_idx = np.random.choice(len(self.trajs), 1, replace=False)[0]
                x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]

                steps = np.random.choice(range(min_steps, max_steps))
                # Subsampling from a trajectory
                if len(x_traj[0]) > steps:
                    ptr = np.random.randint(len(x_traj[0]) - steps)
                    x_slice = slice(ptr, ptr + steps)
                else:
                    x_slice = slice(len(x_traj[1]))

                D.append((x_traj[0][x_slice],
                          sigmoid(sigmoid_params, self.trajs[x_idx][0]) / len(x_traj[1]) * len(x_traj[0][x_slice]) * 10)
                         )

            return D

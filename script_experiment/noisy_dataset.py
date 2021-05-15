import argparse
import os
import pickle

import gym
import joblib
import matplotlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from inverse_rl.models.airl_state import AIRL
from global_utils.utils import set_seed

matplotlib.use('agg')
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (5, 4),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)
from matplotlib import pyplot as plt

from script_experiment.behavior_cloning import Policy


################################
class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()  # [None]


def gen_traj(env, agent, min_length):
    obs, actions, rewards = [env.reset()], [], []
    while True:
        action = agent.act(obs[-1], None, None)
        ob, reward, done, _ = env.step(action)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)

        if done:
            if len(obs) < min_length:
                obs.pop()
                obs.append(env.reset())
            else:
                obs.pop()
                break

    return np.stack(obs, axis=0), np.array(actions), np.array(rewards)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=0.033, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


################################

class NoiseInjectedPolicy(object):
    def __init__(self, env, policy, action_noise_type, noise_level):
        self.action_space = env.action_space
        self.policy = policy
        self.action_noise_type = action_noise_type

        if action_noise_type == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mu=mu, sigma=std)
        elif action_noise_type == 'ou':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=std)
        elif action_noise_type == 'epsilon':
            self.epsilon = noise_level
        else:
            assert False, "no such action noise type: %s" % (action_noise_type)

    def act(self, obs, reward, done):
        if self.action_noise_type == 'epsilon':
            if np.random.random() < self.epsilon:
                return self.action_space.sample()
            else:
                if isinstance(self.policy, Policy):
                    act = self.policy.act(obs, reward, done)  # D-REX Behavior Cloning
                else:
                    act = self.policy.get_action(obs)[0]  # AIRL Policy
        else:
            act = self.policy.act(obs, reward, done)
            act += self.action_noise()

        return np.clip(act, self.action_space.low, self.action_space.high)

    def reset(self):
        self.action_noise.reset()


################################
class BCNoisePreferenceDataset(object):
    def __init__(self, env, max_steps=None, min_margin=None, airl=False):
        self.env = env

        self.max_steps = max_steps
        self.min_margin = min_margin
        self.airl = airl

    def prebuild(self, agent, noise_range, num_trajs, min_length, logdir):
        trajs = []
        for noise_level in tqdm(noise_range):
            noisy_policy = NoiseInjectedPolicy(self.env, agent, 'epsilon', noise_level)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs, _, _, _ in agent_trajs]) < min_length) or \
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs, actions, rewards = gen_traj(self.env, noisy_policy, -1)
                agent_trajs.append((obs, actions, rewards))

            trajs.append((noise_level, agent_trajs))

        self.trajs = trajs

        if not os.path.exists(logdir):
            os.makedirs(logdir)
        print("before dump")
        with open(os.path.join(logdir, 'prebuilt.pkl'), 'wb') as f:
            pickle.dump(self.trajs, f)
        print("after dump")

    def load_prebuilt(self, fname):
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.trajs = pickle.load(f)
            return True
        else:
            return False

    def draw_fig(self, log_dir, demo_trajs, irl_model):
        demo_returns = [np.sum(rewards) for _, _, rewards in demo_trajs]
        demo_ave, demo_std = np.mean(demo_returns), np.std(demo_returns)

        noise_levels = [noise for noise, _ in self.trajs]
        returns = np.array([[np.sum(rewards) for _, _, rewards in agent_trajs] for _, agent_trajs in self.trajs])

        random_agent = RandomAgent(self.env.action_space)
        random_returns = [np.sum(gen_traj(self.env, random_agent, -1)[2]) for _ in range(20)]
        random_ave, random_std = np.mean(random_returns), np.std(random_returns)

        from_to = [np.min(noise_levels), np.max(noise_levels)]

        plt.figure()
        plt.fill_between(from_to,
                         [demo_ave - demo_std, demo_ave - demo_std], [demo_ave + demo_std, demo_ave + demo_std],
                         alpha=0.3)
        plt.plot(from_to, [demo_ave, demo_ave], label='demos')

        plt.fill_between(noise_levels,
                         np.mean(returns, axis=1) - np.std(returns, axis=1),
                         np.mean(returns, axis=1) + np.std(returns, axis=1), alpha=0.3)
        plt.plot(noise_levels, np.mean(returns, axis=1), '-.', label="bc")

        # plot the average of pure noise in dashed line for baseline
        plt.fill_between(from_to,
                         [random_ave - random_std, random_ave - random_std],
                         [random_ave + random_std, random_ave + random_std], alpha=0.3)
        plt.plot(from_to, [random_ave, random_ave], '--', label='random')

        plt.legend(loc="best")
        plt.xlabel("Epsilon")
        # plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "degredation_plot.png"))
        plt.close()

        airl_rewards = []
        for _, agent_trajs in self.trajs:
            airl_rewards.append([])
            for obs, _, _ in agent_trajs:
                airl_rewards[-1].append(np.sum(sess.run(irl_model.reward, feed_dict={
                    irl_model.obs_t: obs
                })))
        airl_rewards = np.array(airl_rewards)

        # AIRL reward corr
        plt.figure()
        plt.scatter(returns.flatten(), airl_rewards.flatten())
        plt.savefig(os.path.join(log_dir, "AIRL_reward_corr.png"))
        plt.close()

        # sigmoid curve regression
        import scipy.optimize

        fig, ax = plt.subplots(nrows=1, ncols=1)

        def sigmoid(p, x):
            x0, y0, c, k = p
            y = c / (1 + np.exp(-k * (x - x0))) + y0
            return y

        def residuals(p, x, y):
            return y - sigmoid(p, x)

        def resize(arr, lower=0.0, upper=1.0):
            arr = arr.copy()
            if lower > upper:
                lower, upper = upper, lower
            arr -= arr.min()
            arr *= (upper - lower) / arr.max()
            arr += lower
            return arr

        # raw data
        x = np.array(noise_levels)
        y = np.mean(airl_rewards, axis=1)
        y = resize(y, lower=0.0)
        print("noise", x)
        print("mean_performance", y)
        p_guess = (np.median(x), np.median(y), 1.0, -1.0)
        p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
            residuals, p_guess, args=(x, y), full_output=1)

        x0, y0, c, k = p
        print('''\
        x0 = {x0}
        y0 = {y0}
        c = {c}
        k = {k}
        '''.format(x0=x0, y0=y0, c=c, k=k))
        print(list(p))
        with open(os.path.join(log_dir, 'fitted_sigmoid_param.pkl'), 'wb') as f:
            pickle.dump(list(p), f)

        xp = np.linspace(0, 1.1, 1500)
        pxp = sigmoid(p, xp)

        # Plot the results
        ax.plot(x, y, '.', xp, pxp, '-')
        ax.set_xlabel('noise')
        ax.set_ylabel('reward', rotation='horizontal')
        ax.grid(True)
        fig.savefig(os.path.join(log_dir, "AIRL_reward_sigmoid.png"))
        plt.close()

    def sample(self, num_samples, args, include_action=False):
        D = []
        acc = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        set_seed(args.seed)
        with tf.Session(config=config) as sess:
            current_num_samples = 0
            while current_num_samples < num_samples:
                # Pick Two Noise Level Set
                x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)
                while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                    x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

                # Pick trajectory from each set
                x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
                y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

                # Subsampling from a trajectory
                if len(x_traj[0]) > self.max_steps:
                    ptr = np.random.randint(len(x_traj[0]) - self.max_steps)
                    x_slice = slice(ptr, ptr + self.max_steps)
                else:
                    x_slice = slice(len(x_traj[1]))

                if len(y_traj[0]) > self.max_steps:
                    ptr = np.random.randint(len(y_traj[0]) - self.max_steps)
                    y_slice = slice(ptr, ptr + self.max_steps)
                else:
                    y_slice = slice(len(y_traj[0]))

                current_num_samples += 1

                if (np.sum(x_traj[2][x_slice]) > np.sum(y_traj[2][y_slice])) == (
                        self.trajs[x_idx][0] < self.trajs[y_idx][0]):
                    acc += 1

                # Done!
                if include_action:
                    D.append((np.concatenate((x_traj[0][x_slice], x_traj[1][x_slice]), axis=1),
                              np.concatenate((y_traj[0][y_slice], y_traj[1][y_slice]), axis=1),
                              0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                             )
                else:
                    D.append((x_traj[0][x_slice],
                              y_traj[0][y_slice],
                              0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                             )
        acc /= num_samples
        print("****************")
        print("acc:", acc)
        print("****************")
        return D


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--log_dir', required=True, help='log dir')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--bc_agent', type=str)
    parser.add_argument('--demo_trajs', required=True,
                        help='suboptimal demo trajectories used for bc (used to generate a figure)')
    parser.add_argument('--airl', action='store_true')
    parser.add_argument('--airl_path', type=str)
    # Noise Injection Hyperparams
    parser.add_argument('--noise_range', default='np.arange(0.,1.0,0.05)',
                        help='decide upto what learner stage you want to give')
    parser.add_argument('--num_trajs', default=5, type=int, help='number of trajectory generated by each agent')
    parser.add_argument('--min_length', default=0, type=int,
                        help='minimum length of trajectory generated by each agent')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    assert args.airl or (not args.airl and args.bc_agent is not None)
    assert args.airl_path is not None
    set_seed(args.seed)

    # Generate a Noise Injected Trajectories
    if args.env_id == "Hopper-v3" or args.env_id == "Ant-v3":
        env = gym.envs.make(args.env_id, terminate_when_unhealthy=False)
    else:
        env = gym.make(args.env_id)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        data = joblib.load(args.airl_path)
        airl_env = data["env"]
        irl_model = AIRL(airl_env, None, state_only=True)
        irl_model.set_params(data['irl_params'])
        policy = data['policy']
        policy.set_param_values(data['policy_params'])
        if args.airl:
            dataset = BCNoisePreferenceDataset(env, airl=True)
            dataset.prebuild(policy, eval(args.noise_range), args.num_trajs, args.min_length, args.log_dir)
        else:
            dataset = BCNoisePreferenceDataset(env)
            agent = Policy(env)
            agent.load(args.bc_agent)
            dataset.prebuild(agent, eval(args.noise_range), args.num_trajs, args.min_length, args.log_dir)

        with open(args.demo_trajs, 'rb') as f:
            demo_trajs = pickle.load(f)

        dataset.draw_fig(args.log_dir, demo_trajs, irl_model)

import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
import csv

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
from tqdm import tqdm

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from utils.utils import StoreDict

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('-e', help='number of episodes', default=5,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    args = parser.parse_args()

    env_id = args.env
    algo = args.algo
    folder = args.folder

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    if env_id == "HalfCheetah-v3":
        model_path = "{}/{}".format(log_path, "rl_model_1000000_steps.zip")
        model_itr = 1000000
    else:
        model_path = "{}/{}".format(log_path, "rl_model_2000000_steps.zip")
        model_itr = 2000000

    set_global_seeds(args.seed)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
    hyperparams["env_wrapper"] = None

    log_dir = args.reward_log if args.reward_log != '' else None

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    env = create_test_env(env_id, n_envs=1, is_atari=False,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    deterministic = True
    result_file = open("{}/{}".format(log_path, "test_real_reward_result.csv"), mode='w')
    fieldnames = ['iteration', 'reward']
    result_writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    result_writer.writeheader()

    if True:
    # for name in tqdm(os.listdir(log_path)):
    #     if name.find("rl_model") == 0:
    #         model_path = "{}/{}".format(log_path, name)
    #     else:
    #         continue
    #     temp = name.split("_")
    #     model_itr = int(temp[2])
        episode_rewards, episode_lengths = [], []
        model = ALGOS[algo].load(model_path, env=env)
        for ep in range(args.e):
            obs = env.reset()
            if not args.no_render:
                env.render('human')
            episode_reward = 0.0
            ep_len = 0
            state = None
            for step in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                # Clip Action to avoid out of bound errors
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, infos = env.step(action)
                if not args.no_render:
                    env.render('human')

                episode_reward += reward[0]
                ep_len += 1

                if done and args.verbose > 0:
                    state = None
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    break
                    episode_reward = 0.0
                    ep_len = 0

        print("%s model test_real_reward itr_%d.mp4" % (args.env, model_itr), np.mean(episode_rewards), np.mean(episode_lengths))
        result_writer.writerow({'iteration': model_itr, 'reward': np.mean(episode_rewards)})
        result_file.flush()

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))


if __name__ == '__main__':
    main()

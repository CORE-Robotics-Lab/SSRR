# adjusted from rl-baselines-zoo:  https://github.com/araffin/rl-baselines-zoo

import os
import time
import uuid
import difflib
import argparse
import importlib
import warnings
from pprint import pprint
from collections import OrderedDict

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import yaml
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.schedules import constfn
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.common.base_class import _UnvecWrapper

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.utils import StoreDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',
                        default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=-1, type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
                        help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
                        help='Ensure that the run has a unique ID')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
                        help='Optional keyword argument to pass to the env constructor')
    args = parser.parse_args()

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = '_{}'.format(uuid.uuid4()) if args.uuid else ''
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1)

    set_global_seeds(args.seed)

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

    print("=" * 10, env_id, "=" * 10)
    print("Seed: {}".format(args.seed))

    # Load hyperparameters from yaml file
    with open('script_experiment/rl_utils/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo
    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if algo_ in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

    n_timesteps = int(hyperparams['n_timesteps'])

    # Convert to python object if needed
    if 'policy_kwargs' in hyperparams.keys() and isinstance(hyperparams['policy_kwargs'], str):
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = "{}/{}/".format(args.log_folder, args.algo)
    save_path = os.path.join(log_path, "{}_{}{}".format(env_id, get_latest_run_id(log_path, env_id) + 1, uuid_str))
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)

    callbacks = []
    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                                            save_path=save_path, name_prefix='rl_model', verbose=1))

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    def create_env(n_envs, eval_env=False):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :return: (Union[gym.Env, VecEnv])
        :return: (gym.Env)
        """
        global hyperparams
        global env_kwargs

        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env else save_path

        env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper, log_dir=log_dir, env_kwargs=env_kwargs)])

        return env

    env = create_env(n_envs)
    # Create test env if needed, do not normalize reward
    eval_env = None
    if args.eval_freq > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        if args.verbose > 0:
            print("Creating test environment")

        eval_callback = EvalCallback(create_env(1, eval_env=True), callback_on_new_best=None,
                                     best_model_save_path=save_path, n_eval_episodes=args.eval_episodes,
                                     log_path=save_path, eval_freq=args.eval_freq)
        callbacks.append(eval_callback)

    if ALGOS[args.algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(args.algo))

    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    print("Log path: {}".format(save_path))

    try:
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass

    # Only save worker of rank 0 when using mpi
    if rank == 0:
        print("Saving to {}".format(save_path))

        model.save("{}/{}".format(save_path, env_id))

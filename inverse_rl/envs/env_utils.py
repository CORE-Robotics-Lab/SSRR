import os

import gym
from gym import Wrapper
from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule, NoVideoSchedule, CappedCubicVideoSchedule, \
    convert_gym_space
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class RllabGymEnv(Env, Serializable):
    def __init__(self, env_name, wrappers=(), wrapper_args=(),
                 record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 post_create_env_seed=None,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        if env_name == "Hopper-v3" or env_name == "Ant-v3":
            env = gym.envs.make(env_name, terminate_when_unhealthy=False)
        else:
            env = gym.envs.make(env_name)
        if post_create_env_seed is not None:
            env.set_env_seed(post_create_env_seed)
        for i, wrapper in enumerate(wrappers):
            if wrapper_args and len(wrapper_args) == len(wrappers):
                env = wrapper(env, **wrapper_args[i])
            else:
                env = wrapper(env)
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def terminate(self):
        if self.monitoring:
            self.env._close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python script_experiment/submit_gym.py %s

    ***************************
                """ % self._log_dir)


class CustomGymEnv(RllabGymEnv):
    def __init__(self, env_name, gym_wrappers=(), wrapper_args=(), record_log=False, record_video=False,
                 post_create_env_seed=None):
        Serializable.quick_init(self, locals())
        self.env_name = env_name
        super(CustomGymEnv, self).__init__(env_name, wrappers=gym_wrappers,
                                           wrapper_args=wrapper_args,
                                           record_log=record_log, record_video=record_video,
                                           post_create_env_seed=post_create_env_seed,
                                           video_schedule=FixedIntervalVideoSchedule(50))

    def _get_obs(self):
        return self.env._get_obs()

    @overrides
    def log_diagnostics(self, paths):
        get_inner_env(self.env).log_diagnostics(paths)

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def get_viewer(self):
        return self.env._get_viewer()


def get_inner_env(env):
    if isinstance(env, ProxyEnv):
        return get_inner_env(env.wrapped_env)
    elif isinstance(env, GymEnv):
        return get_inner_env(env.env)
    elif isinstance(env, Wrapper):
        return get_inner_env(env.env)
    return env

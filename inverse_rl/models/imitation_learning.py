import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils

from inverse_rl.models.architectures import relu_net
from inverse_rl.utils.general import TrainingIterator
from inverse_rl.utils.hyperparametrized import Hyperparametrized
from inverse_rl.utils.math_utils import gauss_log_pdf, categorical_log_pdf

DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'


class ImitationLearning(object, metaclass=Hyperparametrized):
    def __init__(self):
        pass

    def set_demos(self, paths):
        if paths is not None:
            self.expert_trajs = paths

    @staticmethod
    def _compute_path_probs(paths, pol_dist_type=None, insert=True,
                            insert_key='a_logprobs'):
        """
        Returns a N x T matrix of action probabilities
        """
        if insert_key in paths[0]:
            return np.array([path[insert_key] for path in paths])

        if pol_dist_type is None:
            # try to  infer distribution type
            path0 = paths[0]
            if 'log_std' in path0['agent_infos']:
                pol_dist_type = DIST_GAUSSIAN
            elif 'prob' in path0['agent_infos']:
                pol_dist_type = DIST_CATEGORICAL
            else:
                raise NotImplementedError()

        # compute path probs
        Npath = len(paths)
        actions = [path['actions'] for path in paths]
        if pol_dist_type == DIST_GAUSSIAN:
            params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in paths]
            path_probs = [gauss_log_pdf(params[i], actions[i]) for i in range(Npath)]
        elif pol_dist_type == DIST_CATEGORICAL:
            params = [(path['agent_infos']['prob'],) for path in paths]
            path_probs = [categorical_log_pdf(params[i], actions[i]) for i in range(Npath)]
        else:
            raise NotImplementedError("Unknown distribution type")

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    @staticmethod
    def _insert_next_state(paths, pad_val=0.0):
        for path in paths:
            if 'observations_next' in path:
                continue
            nobs = path['observations'][1:]
            nact = path['actions'][1:]
            nobs = np.r_[nobs, pad_val * np.expand_dims(np.ones_like(nobs[0]), axis=0)]
            nact = np.r_[nact, pad_val * np.expand_dims(np.ones_like(nact[0]), axis=0)]
            path['observations_next'] = nobs
            path['actions_next'] = nact
        return paths

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=True):
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32) for key in keys]

    @staticmethod
    def sample_batch(*args, batch_size=32):
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)
        return [data[batch_idxs] for data in args]

    def fit(self, paths, **kwargs):
        raise NotImplementedError()

    def eval(self, paths, **kwargs):
        raise NotImplementedError()

    def _make_param_ops(self, vs):
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        assert len(self._params) > 0
        self._assign_plc = [tf.placeholder(tf.float32, shape=param.get_shape(),
                                           name='assign_%s' % param.name.replace('/', '_').replace(':', '_')) for param
                            in self._params]
        self._assign_ops = [tf.assign(self._params[i], self._assign_plc[i]) for i in range(len(self._params))]

    def get_params(self):
        params = tf.get_default_session().run(self._params)
        assert len(params) == len(self._params)
        return params

    def set_params(self, params):
        tf.get_default_session().run(self._assign_ops, feed_dict={
            self._assign_plc[i]: params[i] for i in range(len(self._params))
        })


class SingleTimestepIRL(ImitationLearning):
    """
    Base class for models that score single timesteps at once
    """

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=False):
        return ImitationLearning.extract_paths(paths, keys=keys, stack=stack)

    @staticmethod
    def unpack(data, paths):
        lengths = [path['observations'].shape[0] for path in paths]
        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx + l])
            idx += l
        return unpacked

    @property
    def score_trajectories(self):
        return False

    def eval_expert_probs(self, expert_paths, policy, insert=False):
        """
        Evaluate expert policy probability under current policy
        """
        for traj in expert_paths:
            if 'agent_infos' in traj:
                del traj['agent_infos']
            if 'a_logprobs' in traj:
                del traj['a_logprobs']

        if isinstance(policy, np.ndarray):
            return self._compute_path_probs(expert_paths, insert=insert)
        elif hasattr(policy, 'recurrent') and policy.recurrent:
            policy.reset([True] * len(expert_paths))
            expert_obs = self.extract_paths(expert_paths, keys=('observations',), stack=True)[0]
            agent_infos = []
            for t in range(expert_obs.shape[1]):
                a, infos = policy.get_actions(expert_obs[:, t])
                agent_infos.append(infos)
            agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
            for key in agent_infos_stack:
                agent_infos_stack[key] = np.transpose(agent_infos_stack[key], axes=[1, 0, 2])
            agent_infos_transpose = tensor_utils.split_tensor_dict_list(agent_infos_stack)
            for i, path in enumerate(expert_paths):
                path['agent_infos'] = agent_infos_transpose[i]
        else:
            for path in expert_paths:
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
        return self._compute_path_probs(expert_paths, insert=insert)

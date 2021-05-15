import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator


class NoisyAIRL(SingleTimestepIRL):
    """


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """

    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='airl'):
        super(NoisyAIRL, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistr(500, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.max_itrs = max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.noise_t = tf.placeholder(tf.float32, [None], name="noise")
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim'):
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)
                with tf.variable_scope('reward'):
                    self.reward = reward_arch(rew_input, dout=1, **reward_arch_args)

                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1)
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(self.obs_t, dout=1)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * fitted_value_fn_n
                log_p_tau = self.reward + self.gamma * fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)
            expanded_noise_t = tf.expand_dims(self.noise_t, axis=1)
            self.loss = -tf.reduce_mean(
                self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq) * (tf.exp(log_q_tau - tf.log(
                        expanded_noise_t * 0.5 ** self.dU + (1 - expanded_noise_t) * tf.exp(log_q_tau) + 1e-37))))
            self.discriminator_predict = tf.cast(log_p_tau > log_q_tau, tf.float32)
            self.discriminator_acc = tf.reduce_mean(self.discriminator_predict * self.labels +
                                                    (1 - self.discriminator_predict) * (1 - self.labels))

            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self._make_param_ops(_vs)

    def fit(self, paths, policy=None, batch_size=32, logger=None, lr=1e-3, **kwargs):
        all_paths = paths

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths + old_paths

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(all_paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs, noises = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs', "noise"))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs, expert_noises = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs', 'noise'))

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch, noise_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, noises, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch, expert_noise_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, expert_noises,
                                  batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(
                np.float32)
            noise_batch = np.concatenate([noise_batch, expert_noise_batch], axis=0)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr,
                self.noise_t: noise_batch
            }

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict=feed_dict)
            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            received_learned_reward, \
            learned_value_function, \
            discrim_output, \
            discrim_acc, \
            discrim_loss = \
                tf.get_default_session().run([self.reward,
                                              self.value_fn,
                                              self.discrim_output,
                                              self.discriminator_acc,
                                              self.loss],
                                             feed_dict={self.act_t: acts,
                                                        self.obs_t: obs,
                                                        self.nobs_t: obs_next,
                                                        self.nact_t: acts_next,
                                                        self.labels: np.zeros((acts.shape[0], 1)),
                                                        self.lprobs: np.expand_dims(path_probs, axis=1),
                                                        self.noise_t: noises})
            with logger.tabular_prefix("AIRL_owntau | "):
                logger.record_tabular('mean_learned_value_function', np.mean(learned_value_function))
                logger.record_tabular('mean_received_learned_reward', np.mean(received_learned_reward))
                logger.record_tabular('mean_log_Q', np.mean(path_probs))
                logger.record_tabular('median_log_Q', np.median(path_probs))
                logger.record_tabular('mean_discrim_output', np.mean(discrim_output))
                logger.record_tabular('mean_discrim_acc', discrim_acc)
                logger.record_tabular('discrim_loss', discrim_loss)

            received_learned_reward, \
            learned_value_function, \
            discrim_output, \
            discrim_acc, \
            discrim_loss = \
                tf.get_default_session().run([self.reward,
                                              self.value_fn,
                                              self.discrim_output,
                                              self.discriminator_acc,
                                              self.loss],
                                             feed_dict={self.act_t: expert_acts,
                                                        self.obs_t: expert_obs,
                                                        self.nobs_t: expert_obs_next,
                                                        self.nact_t: expert_acts_next,
                                                        self.labels: np.ones((expert_acts.shape[0], 1)),
                                                        self.lprobs: np.expand_dims(expert_probs, axis=1),
                                                        self.noise_t: expert_noises})
            with logger.tabular_prefix("AIRL_experttau | "):
                logger.record_tabular('mean_learned_value_function', np.mean(learned_value_function))
                logger.record_tabular('mean_received_learned_reward', np.mean(received_learned_reward))
                logger.record_tabular('mean_log_Q', np.mean(expert_probs))
                logger.record_tabular('median_log_Q', np.median(expert_probs))
                logger.record_tabular('mean_discrim_output', np.mean(discrim_output))
                logger.record_tabular('mean_discrim_acc', discrim_acc)
                logger.record_tabular('discrim_loss', discrim_loss)
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs, noises = self.extract_paths(paths, keys=(
                'observations', 'observations_next', 'actions', 'a_logprobs', 'noise'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.discrim_output,
                                                  feed_dict={self.act_t: acts, self.obs_t: obs,
                                                             self.noise_t: noises,
                                                             self.nobs_t: obs_next,
                                                             self.lprobs: path_probs})
            score = np.log(scores) - np.log(1 - scores)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = tf.get_default_session().run(self.reward,
                                                  feed_dict={self.act_t: acts, self.obs_t: obs})
            score = reward[:, 0]
        return self.unpack(score, paths)

    def eval_single(self, obs, acts):
        reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.obs_t: obs,
                                                         self.act_t: acts})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
                                                       self.qfn],
                                                      feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
            'qfn': qfn,
        }

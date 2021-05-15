import argparse
import pickle

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.samplers.noisy_sampler import NoisySampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from global_utils.utils import *
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--noisy', action='store_true')
    args = parser.parse_args()
    set_seed(0)

    env = TfEnv(CustomGymEnv('Hopper-v3', record_video=False, record_log=False))

    with open("./demos/suboptimal_demos/hopper/dataset.pkl", "rb") as f:
        experts = pickle.load(f)
        processed_experts = []
        for expert in experts:
            processed_experts.append({
                'observations': expert[0],
                'actions': expert[1]
            })
            print("demo reward", np.sum(expert[2]))
        experts = processed_experts

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 1000
    log_path = args.output_dir

    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True, fusion=True, max_itrs=discriminator_update_step,
                     score_discrim=False)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    if args.noisy:
        print("Using Noisy Sampler")
        sampler_class = NoisySampler
    else:
        print("Using Normal Sampler")
        sampler_class = VectorizedSampler
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=n_epochs,
        batch_size=10000,
        max_path_length=1000,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=discriminator_update_step,
        irl_model_wt=1.0,
        entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        sampler_cls=sampler_class,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        step_size=0.005,  # max_kl for TRPO
        optimizer_args=dict(reg_coeff=0.1, cg_iters=15)
    )

    if 'test' in log_path and os.path.exists(log_path):
        import shutil
        shutil.rmtree(log_path)
    assert not os.path.exists(log_path), "log path already exist! "
    with rllab_logdir(algo=algo, dirname=log_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            algo.train()


if __name__ == "__main__":
    main()

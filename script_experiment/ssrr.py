import argparse
import pickle
from pathlib import Path

import gym

from Agents.SSRRAgent import SSRRAgent
from Datasets.NoiseDataset import NoiseDataset
from global_utils.utils import *


def train_reward(args):
    log_dir = Path(args.log_dir)

    with open(str(log_dir / 'args.txt'), 'w') as f:
        f.write(str(args))

    env = gym.make(args.env_id)

    ob_dims = env.observation_space.shape[-1]
    ac_dims = env.action_space.shape[-1]

    dataset = NoiseDataset(env)

    loaded = dataset.load_prebuilt(args.noise_injected_trajs)
    assert loaded

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d' % i):
            model = SSRRAgent(include_action=args.include_action,
                              ob_dim=ob_dims,
                              ac_dim=ac_dims,
                              layers=[256, 256],
                              batch_size=64)
            models.append(model)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    with open(args.sigmoid_params_path, 'rb') as f:
        p = pickle.load(f)

    for i, model in enumerate(models):
        D = dataset.sample_not_paired(args.D, args.min_steps, args.max_steps, p, include_action=args.include_action)

        model.train(D, iter=args.iter, l2_reg=args.l2_reg, debug=True)

        model.saver.save(sess, os.path.join(str(log_dir), 'model_%d.ckpt' % (i)), write_meta_graph=False)

    sess.close()


def eval_reward(args):
    env = gym.make(args.env_id)

    ob_dims = env.observation_space.shape[-1]
    ac_dims = env.action_space.shape[-1]

    dataset = NoiseDataset(env)

    loaded = dataset.load_prebuilt(args.noise_injected_trajs)
    assert loaded

    # Load Seen Trajs
    seen_trajs = [
        (obs, actions, rewards) for _, trajs in dataset.trajs for obs, actions, rewards in trajs
    ]

    # Load Unseen Trajectories
    if args.unseen_trajs:
        with open(args.unseen_trajs, 'rb') as f:
            unseen_trajs = pickle.load(f)
    else:
        uneen_trajs = []

    # Load Demo Trajectories used for BC
    with open(args.bc_trajs, 'rb') as f:
        bc_trajs = pickle.load(f)

    graph = tf.Graph()
    config = tf.ConfigProto()  # Run on CPU
    config.gpu_options.allow_growth = True

    with graph.as_default():
        set_seed(args.seed)
        models = []
        for i in range(args.num_models):
            with tf.variable_scope('model_%d' % i):
                model = SSRRAgent(include_action=args.include_action,
                                  ob_dim=ob_dims,
                                  ac_dim=ac_dims,
                                  layers=[256, 256],
                                  batch_size=64)
                models.append(model)

    sess = tf.Session(graph=graph, config=config)
    set_seed(args.seed)
    for i, model in enumerate(models):
        with sess.as_default():
            model.saver.restore(sess, os.path.join(args.log_dir, 'model_%d.ckpt' % i))

    # Calculate Predicted Returns
    def _get_return(obs, acs):
        with sess.as_default():
            return np.sum([model.get_reward(obs, acs) for model in models]) / len(models)

    seen = [1] * len(seen_trajs) + [0] * len(unseen_trajs) + [2] * len(bc_trajs)
    gt_returns, pred_returns = [], []

    for obs, actions, rewards in seen_trajs + unseen_trajs + bc_trajs:
        gt_returns.append(np.sum(rewards))
        pred_returns.append(_get_return(obs, actions))
    sess.close()
    print(np.corrcoef(gt_returns, pred_returns))
    print(np.corrcoef(gt_returns, pred_returns)[0, 1])

    # Draw Result
    def _draw(gt_returns, pred_returns, seen, figname=False):
        """
        gt_returns: [N] length
        pred_returns: [N] length
        seen: [N] length
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pylab
        from matplotlib import pyplot as plt

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('ggplot')
        params = {
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'legend.fontsize': 'xx-large',
            # 'figure.figsize': (6, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize': 'xx-large',
            'xtick.labelsize': 'xx-large',
            'ytick.labelsize': 'xx-large'}
        matplotlib.pylab.rcParams.update(params)

        def _convert_range(x, minimum, maximum, a, b):
            return (x - minimum) / (maximum - minimum) * (b - a) + a

        def _no_convert_range(x, minimum, maximum, a, b):
            return x

        convert_range = _convert_range
        # convert_range = _no_convert_range

        gt_max, gt_min = max(gt_returns), min(gt_returns)
        pred_max, pred_min = max(pred_returns), min(pred_returns)
        max_observed = np.max(gt_returns[np.where(seen != 1)])

        # Draw P
        fig, ax = plt.subplots()

        ax.plot(gt_returns[np.where(seen == 0)],
                [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 0)]],
                'go')  # unseen trajs
        ax.plot(gt_returns[np.where(seen == 1)],
                [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 1)]],
                'bo')  # seen trajs for T-REX
        ax.plot(gt_returns[np.where(seen == 2)],
                [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 2)]],
                'ro')  # seen trajs for BC

        ax.plot([gt_min - 5, gt_max + 5], [gt_min - 5, gt_max + 5], 'k--')
        ax.set_xlabel("Ground Truth Returns")
        ax.set_ylabel("Predicted Returns (normalized)")
        fig.tight_layout()

        plt.savefig(figname)
        plt.close()

    save_path = os.path.join(args.log_dir, 'gt_vs_pred_rewards.png')
    _draw(np.array(gt_returns), np.array(pred_returns), np.array(seen), save_path)


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--mode', default='train_reward',
                        choices=['all', 'train_reward', 'eval_reward', 'train_rl', 'eval_rl'])
    parser.add_argument('--sigmoid_params_path', type=str, required=True)
    # Args for T-REX
    ## Dataset setting
    parser.add_argument('--noise_injected_trajs', default='')
    parser.add_argument('--unseen_trajs', default='', help='used for evaluation only')
    parser.add_argument('--bc_trajs', default='', help='used for evaluation only')
    parser.add_argument('--D', default=5000, type=int, help='|D| in the preference paper')
    parser.add_argument('--min_steps', default=50, type=int, help='minimum length of subsampled trajecotry')
    parser.add_argument('--max_steps', default=51, type=int, help='maximum length of subsampled trajecotry')
    parser.add_argument('--min_noise_margin', default=0.3, type=float, help='')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    ## Network setting
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.0, type=float,
                        help='noise level to add on training label (another regularization)')
    parser.add_argument('--iter', default=3000, type=int, help='# trainig iters')
    # Args for PPO
    parser.add_argument('--rl_runs', default=3, type=int)
    parser.add_argument('--num_timesteps', default=int(1e6), type=int)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == 'train_reward':
        train_reward(args)
        tf.reset_default_graph()
        set_seed(args.seed)
        eval_reward(args)
    elif args.mode == 'eval_reward':
        eval_reward(args)
    else:
        assert False

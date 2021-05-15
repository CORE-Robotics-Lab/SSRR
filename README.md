# Self-Supervised Reward Regression (SSRR)

Codebase for CoRL 2021 paper "[Learning from Suboptimal Demonstration via Self-Supervised Reward Regression
](https://corlconf.github.io/corl2021/paper_281/)"
Authors: [Letian "Zac" Chen](https://core-robotics.gatech.edu/people/zac-chen/), [Rohan Paleja](https://core-robotics.gatech.edu/people/rohan-paleja/), [Matthew Gombolay](https://core-robotics.gatech.edu/people/matthew-gombolay/)

## Usage
### Quick overview
The pipeline of SSRR includes 
1. Initial IRL: Noisy-AIRL or AIRL.
2. Noisy Dataset Generation: use initial policy learned in step 1 to generate trajectories with different noise levels and criticize trajectories with initial reward. 
3. Sigmoid Fitting: fit a sigmoid function for the noise-performance relationship using the data obtained in step 2. 
4. Reward Learning: learn a reward function by regressing to the sigmoid relationship obtained in step 3. 
5. Policy Learning: learn a policy by optimizing the reward learned in step 4. 

I know this is a long README, but please make sure you read the entirety before trying out our code. Trust me, that will save your time! 

### Dependencies and Environment Preparations
Code is tested with Python 3.6 with Anaconda.

Required packages:
```bash
pip install scipy path.py joblib==0.12.3 flask h5py matplotlib scikit-learn pandas pillow pyprind tqdm nose2 mujoco-py cached_property cloudpickle git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano git+https://github.com/neocxi/Lasagne.git@484866cf8b38d878e92d521be445968531646bb8#egg=Lasagne plotly==2.0.0 gym[all]==0.14.0 progressbar2 tensorflow-gpu==1.15 imgcat
```

Test sets of trajectories could be downloaded at [Google Drive](https://drive.google.com/drive/folders/1uzAwV-nzL4uSHO2opMjdEXA1bQrKvXkO?usp=sharing) because Github could not hold files that are larger than 100MB! After downloading, please put `full_demos/` under `demos/`.

If you are directly running python scripts, you will need to add the project root and the rllab_archive folder into your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/this/repo/:/path/to/this/repo/rllab_archive/
```

If you are using the bash scripts provided (for example, `noisy_airl_ssrr_drex_comparison_halfcheetah.sh`), make sure to replace the first line to be
```bash
export PYTHONPATH=/path/to/this/repo/:/path/to/this/repo/rllab_archive/
```

### Initial IRL
We provide code for AIRL and Noisy-AIRL implementation. 

#### Running
Examples of running command would be
```bash
python script_experiment/halfcheetah_airl.py --output_dir=./data/halfcheetah_airl_test_1
python script_experiment/hopper_noisy_airl.py --output_dir=./data/hopper_noisy_airl_test_1 --noisy
```
Please note for Noisy-AIRL, you have to include the `--noisy` flag to make it actually sample trajectories with noise, otherwise it only changes the loss function according to Equation 6 in the paper. 

#### Results
The result will be available in the output dir specified, and we recommend using [rllab viskit](https://github.com/rll/rllab/tree/master/rllab/viskit) to visualize it.

We also provide our run results available in `data/{halfcheetah/hopper/ant}_{airl/noisy_airl}_test_1` if you want to skip this step! 

#### Code Structure
The AIRL and Noisy-AIRL codes reside in `inverse_rl/` with rllab dependencies in `rllab_archive`. 
The AIRL code is adjusted from the original AIRL codebase [https://github.com/justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl).
The rllab archive was adjusted from the original rllab codebase [https://github.com/rll/rllab](https://github.com/rll/rllab). 

### Noisy Dataset Generation & Sigmoid Fitting
We implemented noisy dataset generation and sigmoid fitting together in code.  

#### Running
Examples of running command would be 
```bash
python script_experiment/noisy_dataset.py \
   --log_dir=./results/halfcheetah/temp/noisy_dataset/ \
   --env_id=HalfCheetah-v3 \
   --bc_agent=./results/halfcheetah/temp/bc/model.ckpt \
   --demo_trajs=./demos/suboptimal_demos/ant/dataset.pkl \
   --airl_path=./data/halfcheetah_airl_test_1/itr_999.pkl \
   --airl \
   --seed="${loop}"
```

Note that flag `--airl` determines whether we utilize the `--airl_path` or `--bc_agent` policy to generate the trajectory. Therefore, `--bc_agent` is optional when `--airl` present. For behavior cloning policy, please refer to [https://github.com/dsbrown1331/CoRL2019-DREX](https://github.com/dsbrown1331/CoRL2019-DREX).

The `--airl_path` always provide the initial reward to criticize the generated trajectories no matter whether `--airl` present.

#### Results
The result will be available in the log dir specified. 

We also provide our run results available in `results/{halfcheetah/hopper/ant}/{airl/noisy_airl}_data_ssrr_{1/2/3/4/5}/noisy_dataset/` if you want to skip this step!

#### Code Structure
Noisy dataset generation and Sigmoid fitting are implemented in `script_experiment/noisy_dataset.py`. 

### Reward Learning
We provide SSRR and D-REX implementation. 

#### Running
Examples of running command would be 
```bash
  python script_experiment/drex.py \
   --log_dir=./results/halfcheetah/temp/drex \
   --env_id=HalfCheetah-v3 \
   --bc_trajs=./demos/suboptimal_demos/halfcheetah/dataset.pkl \
   --unseen_trajs=./demos/full_demos/halfcheetah/unseen_trajs.pkl \
   --noise_injected_trajs=./results/halfcheetah/temp/noisy_dataset/prebuilt.pkl \
   --seed="${loop}"
  python script_experiment/ssrr.py \
   --log_dir=./results/halfcheetah/temp/ssrr \
   --env_id=HalfCheetah-v3 \
   --mode=train_reward \
   --noise_injected_trajs=./results/halfcheetah/temp/noisy_dataset/prebuilt.pkl \
   --bc_trajs=demos/suboptimal_demos/halfcheetah/dataset.pkl \
   --unseen_trajs=demos/full_demos/halfcheetah/unseen_trajs.pkl \
   --min_steps=50 --max_steps=500 --l2_reg=0.1 \
   --sigmoid_params_path=./results/halfcheetah/temp/noisy_dataset/fitted_sigmoid_param.pkl \
   --seed="${loop}"
```

The bash script also helps combining running of noisy dataset generation, sigmoid fitting, and reward learning, and repeats several times:
```bash
./airl_ssrr_drex_comparison_halfcheetah.sh
```

#### Results
The result will be available in the log dir specified.

The correlation between the predicted reward and the ground-truth reward tested on the unseen_trajs is reported at the end of running on console, or, if you are using the bash script, at the end of the d_rex.log or ssrr.log.

We also provide our run results available in `results/{halfcheetah/hopper/ant}/{airl/noisy_airl}_data_ssrr_{1/2/3/4/5}/{drex/ssrr}/`. 

#### Code Structure
SSRR is implemented in `script_experiment/ssrr.py`, `Agents/SSRRAgent.py`, `Datasets/NoiseDataset.py`. 

D-REX is implemented in `script_experiment/drex.py`, `scrip_experiment/drex_utils.py`, and `script_experiment/tf_commons/ops`. 

Both implementations are adapted from [https://github.com/dsbrown1331/CoRL2019-DREX](https://github.com/dsbrown1331/CoRL2019-DREX). 

### Policy Learning
We utilize [stable-baselines](https://github.com/hill-a/stable-baselines) to optimize policy over the reward we learned. 

#### Running
Before running, you should edit `script_experiment/rl_utils/sac.yml` to change the learned reward model directory, for example:
```bash
  env_wrapper: {"script_experiment.rl_utils.wrappers.CustomNormalizedReward": {"model_dir": "/home/zac/Programming/Zac-SSRR/results/halfcheetah/noisy_airl_data_ssrr_4/ssrr/", "ctrl_coeff": 0.1, "alive_bonus": 0.0}}
```

Examples of running command would be 
```bash
python script_experiment/train_rl_with_learned_reward.py \
 --algo=sac \
 --env=HalfCheetah-v3 \
 --tensorboard-log=./results/HalfCheetah_custom_reward/ \
 --log-folder=./results/HalfCheetah_custom_reward/ \
 --save-freq=10000
```

Please note the flag `--env-kwargs=terminate_when_unhealthy:False` is necessary for Hopper and Ant as discussed in our paper Supplementary D.1.

Examples of running evaluation the learned policy's ground-truth reward would be 
```bash
python script_experiment/test_rl_with_ground_truth_reward.py \
 --algo=sac \
 --env=HalfCheetah-v3 \
 -f=./results/HalfCheetah_custom_reward/ \
 --exp-id=1 \
 -e=5 \
 --no-render \
 --env-kwargs=terminate_when_unhealthy:False
```

#### Results
The result will be available in the log folder specified.

We also provide our run results in `results/`.  

#### Code Structure
The code `script_experiment/train_rl_with_learned_reward.py` and `utils/` call stable-baselines library to learn a policy with the learned reward function. Note that `utils` could not be renamed because of the rl-baselines-zoo constraint. 

The codes are adjusted from [https://github.com/araffin/rl-baselines-zoo](https://github.com/araffin/rl-baselines-zoo). 

## Random Seeds
Because of the inherent stochasticity of GPU reduction operations such as `mean` and `sum` ([https://github.com/tensorflow/tensorflow/issues/3103](https://github.com/tensorflow/tensorflow/issues/3103)), even if we set the random seed, we cannot reproduce the exact result every time. Therefore, we encourage you to run multiple times to reduce the random effect.

If you have a nice way to get the same result each time, please let us know! 

## Ending Thoughts
We welcome discussions or extensions of our paper and code in Issues!

Feel free to leave a star if you like this repo! 

For more exciting work our lab (CORE Robotics Lab in Georgia Institute of Technology led by Professor Matthew Gombolay), check out our [website](https://core-robotics.gatech.edu/)! 

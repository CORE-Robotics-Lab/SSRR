#!/bin/bash

export PYTHONPATH=/home/zac/Programming/Zac-SSRR/:/home/zac/Programming/Zac-SSRR/rllab_archive/

for loop in {1..5}
do
  echo "LOOP ${loop}"
  rm -rf ./results/halfcheetah/airl_data_ssrr_${loop}
  mkdir -p ./results/halfcheetah/airl_data_ssrr_${loop}

  echo "Noise dataset..."
  mkdir ./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/
  python script_experiment/noisy_dataset.py --log_dir=./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/ \
   --env_id=HalfCheetah-v3 \
   --bc_agent=./results/halfcheetah/airl_data_ssrr_${loop}/bc/model.ckpt \
   --demo_trajs=./demos/suboptimal_demos/halfcheetah/dataset.pkl \
   --airl_path=./data/halfcheetah_airl_test_1/itr_999.pkl \
   --airl \
   --seed="${loop}" \
   > ./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/noise_dataset.log \
   2> ./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/noise_dataset.error

  echo "D-REX..."
  mkdir ./results/halfcheetah/airl_data_ssrr_${loop}/drex/
  python script_experiment/drex.py --log_dir=./results/halfcheetah/airl_data_ssrr_${loop}/drex \
   --env_id=HalfCheetah-v3 \
   --bc_trajs=./demos/suboptimal_demos/halfcheetah/dataset.pkl \
   --unseen_trajs=./demos/full_demos/halfcheetah/unseen_trajs.pkl \
   --noise_injected_trajs=./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/prebuilt.pkl \
   --seed="${loop}" \
   > ./results/halfcheetah/airl_data_ssrr_${loop}/drex/d_rex.log \
   2> ./results/halfcheetah/airl_data_ssrr_${loop}/drex/d_rex.error

  echo "SSRR..."
  mkdir ./results/halfcheetah/airl_data_ssrr_${loop}/ssrr/
  python script_experiment/ssrr.py --log_dir=./results/halfcheetah/airl_data_ssrr_${loop}/ssrr \
   --env_id=HalfCheetah-v3 \
   --mode=train_reward \
   --noise_injected_trajs=./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/prebuilt.pkl \
   --bc_trajs=demos/suboptimal_demos/halfcheetah/dataset.pkl \
   --unseen_trajs=demos/full_demos/halfcheetah/unseen_trajs.pkl \
   --min_steps=50 --max_steps=500 --l2_reg=0.1 \
   --sigmoid_params_path=./results/halfcheetah/airl_data_ssrr_${loop}/noisy_dataset/fitted_sigmoid_param.pkl \
   --seed="${loop}" \
   > ./results/halfcheetah/airl_data_ssrr_${loop}/ssrr/ssrr.log \
   2> ./results/halfcheetah/airl_data_ssrr_${loop}/ssrr/ssrr.error
done
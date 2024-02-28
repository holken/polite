#!/bin/bash
for seed in 1789512682 2477371463 1713025571 2239347892 271041149
do
  python train.py --algo ppo --env HalfCheetah-v3 --n_queries 20 --n_init_queries 20 --max_queries 200 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1 --regularize
done
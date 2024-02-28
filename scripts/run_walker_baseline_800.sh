#!/bin/bash
for seed in 1361082395 420645331 1339675450 2601926218 3196176496
do
  python train.py --algo ppo --env Walker2d-v3 --n_queries 80 --n_init_queries 80 --max_queries 800 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1
done
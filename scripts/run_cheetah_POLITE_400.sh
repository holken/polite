#!/bin/bash
for seed in 3196006235 335280825 1371577540 2295717639 2658490917
do
  python train.py --algo ppo --env HalfCheetah-v3 --n_queries 40 --n_init_queries 40 --max_queries 400 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1  --regularize
done
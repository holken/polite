#!/bin/bash
for seed in 153525426 1818526301 2992013839 3267993059 736833007
do
  python train.py --algo ppo --env HalfCheetah-v3 --n_queries 80 --n_init_queries 80 --max_queries 800 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1
done
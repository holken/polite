#!/bin/bash
for seed in 4098201537 1965646730 1893954428 4081794309 3537201974
do
  python train.py --algo ppo --env HalfCheetah-v3 --n_queries 40 --n_init_queries 40 --max_queries 400 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1
done
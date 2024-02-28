#!/bin/bash
for seed in 1542299285 2074760763 999931978 3174106417 3770989333
do
  python train.py --algo ppo --env Walker2d-v3 --n_queries 40 --n_init_queries 40 --max_queries 400 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1
done
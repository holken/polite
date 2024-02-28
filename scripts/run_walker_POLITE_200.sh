#!/bin/bash
for seed in 1767166233 3968382784 3326175668 1762089001 1358551840
do
  python train.py --algo ppo --env Walker2d-v3 --n_queries 20 --n_init_queries 20 --max_queries 200 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1 --regularize
done
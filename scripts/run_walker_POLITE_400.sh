#!/bin/bash
for seed in 2007306672 3387441270 2106497010 2957207769 3129356124
do
  python train.py --algo ppo --env Walker2d-v3 --n_queries 40 --n_init_queries 40 --max_queries 400 --truth 90 --seed $seed --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" --track --wandb-project-name PrefLearn --wandb-entity sholk --eval-freq -1 --regularize
done
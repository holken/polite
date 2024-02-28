import argparse
import difflib
import importlib
import os
import time
import uuid
from datetime import datetime

import gym as gym26
import gymnasium as gym
import numpy as np
import seaborn
import torch as th
from gym import register
from gym_unity.envs import UnityToGymWrapper
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from stable_baselines3.common.utils import set_random_seed

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.exp_manager import ExperimentManager
from utils.utils import ALGOS, StoreDict, get_latest_run_id

seaborn.set()


def register_custom_envs():
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
    drawer_close_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]
    button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
    register(
        id="door-open-v2",
        entry_point=door_open_goal_observable_cls,
        max_episode_steps=500,
        apply_api_compatibility=True
    )

    register(
        id="drawer-close-v2",
        entry_point=drawer_close_goal_observable_cls,
        max_episode_steps=500,
        apply_api_compatibility=True
    )

    register(
        id="button-press-v2",
        entry_point=button_press_goal_observable_cls,
        max_episode_steps=500,
        apply_api_compatibility=True
    )

    env = UnityToGymWrapper
    register(
        id="Social-Nav-v1",
        entry_point=env,
        apply_api_compatibility=True
    )


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=25000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-total-trials",
        help="Number of (potentially pruned) trials for optimizing hyperparameters. "
        "This applies to the entire optimization process and takes precedence over --n-trials if set.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization."
        "Default is 1 evaluation per 100k timesteps.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument("-uuid", "--uuid", action="store_true", default=False, help="Ensure that the run has a unique ID")
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="sb3", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    parser.add_argument("--regularize", action="store_true", default=False, help="Regularized preference learning")
    parser.add_argument("--workerid", type=int, default=-1)
    parser.add_argument("--n_queries", type=int, default=-1)
    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument("--n_init_queries", type=int, default=-1)
    parser.add_argument("--truth", type=int, default=-1)
    parser.add_argument("--extraname", type=str, default="", help="The name that distinguish in wb")
    parser.add_argument("--prefcallback", type=str, default="", help="override callback")
    args = parser.parse_args()

    register_custom_envs()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.keys())
    registered_envs.update(gym26.envs.registry.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    if args.track:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            )

        currTime = datetime.now()
        date_time = currTime.strftime("%H:%M:%S-%d/%m/%Y")

        wandb.login(key="bc48b46ad36770df58a83389e5609a4aab876af6")
        run_name = f"{args.env}__{date_time}__{args.seed}__{args.extraname}"
        run = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs/{run_name}"

    exp_manager = ExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.max_total_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        device=args.device,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            # we need to save the loaded hyperparameters
            args.saved_hyperparams = saved_hyperparams
            run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
            log_path = f"{args.log_folder}/{args.algo}/"
            specific_log_path = f"{log_path}{env_id}_{get_latest_run_id(log_path, env_id)}{uuid_str}/"
            if wandb:
                wandb.save(f'{specific_log_path}*')
            #run.upload_file(f"{log_path}{env_id}_{get_latest_run_id(log_path, env_id)}{uuid_str}/")
    else:
        exp_manager.hyperparameters_optimization()

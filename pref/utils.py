import math
import pickle as pkl
from datetime import datetime

import gym
import yaml
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def save_pickle(obj, name):
    with open(name + ".pkl", 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name + ".pkl", 'rb') as handle:
        return pkl.load(handle)


def shift_interval(from_a, from_b, to_c, to_d, t):
    shift = to_c + ((to_d - to_c) / (from_b - from_a)) * (t - from_a)
    shift = max(shift, to_c)
    shift = min(shift, to_d)
    return shift


def return_date():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string


def find_model_name(name, path):
    name += ".zip"
    for root, dirs, files in os.walk(path):
        if name in files:
            file_path = os.path.join(root, name)
            return file_path[:-4]  #removing .zip


def moving_average(rewards, last_N):
    scores = []
    averages = []
    stds = []
    for _ in range(len(rewards)):
        scores.append(rewards[_])
        average = sum(scores[-last_N:]) / len(scores[-last_N:])
        averages.append(average)
        x = scores[-last_N:]
        x_mean = averages[-last_N:]
        diff = []
        for _ in range(len(x)):
            diff.append((abs(x[_] - x_mean[_]))**(1 / 2))
        diff_sum = sum(diff)

        stds.append(diff_sum / last_N)

    return averages, stds


def moving_std(rewards, last_N):
    scores = []
    averages = []
    stds = []
    for _ in range(len(rewards)):
        scores.append(rewards[_])
        average = sum(scores[-last_N:]) / len(scores[-last_N:])
        averages.append(average)
        x = scores[-last_N:]
        x_mean = averages[-last_N:]
        diff = []
        for _ in range(len(x)):
            diff.append((abs(x[_] - x_mean[_]))**(1 / 2))
        diff_sum = sum(diff)

        stds.append(diff_sum / last_N)

    return stds


def kl_divergence(u1, sig1, u2, sig2):
    term1 = math.log(sig2 / sig1)
    term2 = (sig1 * sig1 + (u1 - u2) * (u1 - u2)) / (2 * sig2 * sig2)
    term3 = -1 / 2
    return term1 + term2 + term3


def average(lst):
    return sum(lst) / len(lst)


def get_hyperparameters(env_name):
    """
    Loads hyperparameter from the hyperparameters.yml file.
    If the environment does not exist, it will load default hyperparameters.
    :param env_name: Name of the gym environment
    :return: Model, environment, and learn hyperparameters
    """

    with open("hyperparameters.yml", mode="rt", encoding="utf-8") as file:
        content = yaml.safe_load(file)
        if env_name in content:
            hyperparameters = content[env_name]
        else:
            print("No hyperparameters for given environment, loading default")
            hyperparameters = content["Default"]
        env_hyperparameters = dict()
        learn_hyperparameters = dict()
        misc = dict()
        print(hyperparameters)

        if 'n_timesteps' in hyperparameters.keys():
            hyperparameters['n_timesteps'] = int(hyperparameters['n_timesteps'])

        if 'n_envs' in hyperparameters.keys():
            env_hyperparameters['n_envs'] = hyperparameters['n_envs']
            del hyperparameters['n_envs']

        if 'n_timesteps' in hyperparameters.keys():
            learn_hyperparameters['total_timesteps'] = hyperparameters['n_timesteps']
            del hyperparameters['n_timesteps']

        if 'normalize' in hyperparameters.keys():
            misc['normalize'] = hyperparameters['normalize']
            del hyperparameters['normalize']

        if 'policy_kwargs' in hyperparameters.keys():
            #TODO parse dict
            hyperparameters['policy_kwargs'] = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    return hyperparameters, env_hyperparameters, learn_hyperparameters, misc

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                                ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
def get_env_dimensions(env_id):
    tmp_env = gym.make(env_id)
    action_size = tmp_env.action_space.shape[0] if len(tmp_env.action_space.shape) > 0 else 1
    state_size = tmp_env.observation_space.shape
    print(state_size)
    tmp_env.close()
    return action_size, state_size

if __name__ == "__main__":
    import os

    dirs = find_model_name(name="PPO-optimal.zip", path=os.getcwd())

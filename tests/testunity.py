import time

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
# Start interacting with the environment.
print("gummi")
env.reset()
print("hubbabubba")
behavior_names = list(env.behavior_specs.keys())
print(behavior_names)

while True:
    act = ActionTuple(continuous= np.array([[1,0,0]]))
    env.set_actions(behavior_names[0], act)
    env.step()
    normal_step, terminal_step = env.get_steps(behavior_names[0])
    if len(terminal_step) == 0:
        obs = normal_step.obs[0]
        rew = normal_step.reward[0]
        done = False
    else:
        obs = terminal_step.obs[0]
        rew = terminal_step.reward[0]
        done = True
        print("terminal")
        print(obs)
        print(rew)

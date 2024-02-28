import os
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize

#env_id = "Walker2d-v3"
env_id = "HalfCheetah-v3"
policy_map = os.path.join(os.getcwd(), "policies")
policy_name = "entropy_cheetah.zip"
vec_norm_name = "entropy_cheetah_vecnormalize.pkl"
video_folder = "./videos/"
video_length = 3000

policy_path = os.path.join(policy_map, policy_name)
norm_path = os.path.join(policy_map, vec_norm_name)

#env = gym.make(env_id)
env = DummyVecEnv([lambda: gym.make(env_id)])
#env = VecNormalize(env, gamma=0.99, norm_reward=False)
env = VecNormalize.load(norm_path, env)
#env = VecVideoRecorder(env, video_folder,
#                       record_video_trigger=lambda x: x == 1000, video_length=video_length,
#                       name_prefix=f"random-agent-{env_id}")
env.training = False
env.norm_reward = False

if os.path.exists(policy_map):
    print("found")
else:
    print(os.getcwd())


model = PPO.load(policy_path)

# Enjoy trained agent
env.render()
time.sleep(1)
obs = env.reset()
rew = 0
for i in range(5500):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    rew += rewards
    if dones:
        print(rew)
        env.reset()
        rew = 0

    #env.render()

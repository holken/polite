from abc import abstractmethod

import gym
import torch


class HumanReward(gym.Wrapper):
    """
    Environment wrapper to replace the env reward function with a human based reward function.
    In addition, it logs both the env and human reward per episode.
    """

    def __init__(self, env, hc, logger):
        super().__init__(env)
        self.env = env
        self.current_state = []
        self.episode_reward_human = 0
        self.episode_true_reward = 0
        self.t = 0
        self.human_model = hc
        self.logger = logger

    @torch.no_grad()
    def step(self, action):
        observation = self.get_human_reward_observation(action)

        next_state, reward, done, info = self.env.step(action)
        self.episode_true_reward += reward

        reward_human = self.human_model.reward_model(observation)[0].detach().numpy().item()
        self.episode_reward_human += reward_human

        reward = reward_human
        self.current_state = next_state

        if done:
            self.logger.record("rollout/ep_human_rew", self.episode_reward_human)
            self.logger.record("rollout/ep_true_rew_mean", self.episode_true_reward)

            self.episode_reward_human = 0
            self.episode_true_reward = 0

        return next_state, reward, done, info

    def get_human_reward_observation(self, action):
        """
        Creates an observation by concatenating env observations with actions.
        The if handles edge cases that can show up depending on env/vectorization.
        :param action: action that the agent will take given current observation
        :return: observation (obs, act) to be used in a human reward model
        """
        action_tensor = torch.tensor(action)
        if len(action_tensor.shape) != 0:
            observation = torch.cat([torch.tensor(self.current_state), action_tensor])
        else:
            action_tensor = torch.tensor(action)
            action_tensor = torch.unsqueeze(action_tensor, 0)
            observation = torch.cat([torch.tensor(self.current_state), action_tensor])
        observation = torch.tensor(observation)
        observation = observation.type(torch.float32).unsqueeze(0)
        return observation

    def reset(self):
        state = self.env.reset()
        self.current_state = state

        return state

    def observation(self, obs):
        self.current_state = obs
        return obs



import collections
import gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader


def reset_env(env):
    obs = env.reset()
    is_tuple = isinstance(obs, tuple)
    obs = obs[0] if is_tuple else obs
    return obs

# TODO: because gyms support for older environment we don't need this anymore
def env_step(action, env):
    step_obj = env.step(action)
    done = True if step_obj[2] or step_obj[3] else False
    step_obj = (step_obj[0], step_obj[1], done)
    return step_obj


class PreferenceCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,
                 hc,
                 env_name='LunarLanderContinuous-v2',
                 n_queries=10,
                 initial_reward_estimation_epochs=200,
                 reward_training_epochs=50,
                 truth=90,
                 traj_length=50,
                 smallest_rew_threshold=0,
                 largest_rew_threshold=0,
                 n_initial_queries=200,
                 max_queries=1400,
                 verbose=0,
                 seed=12345,
                 workerid=-1,
                 recorder=None):
        super(PreferenceCallback, self).__init__(verbose)
        self.n_queries = n_queries
        self.hc = hc
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.env_name = env_name
        self.reward_training_epochs = reward_training_epochs
        self.truth = truth
        self.traj_length = traj_length
        self.smallest_rew_threshold = smallest_rew_threshold
        self.largest_rew_threshold = largest_rew_threshold
        self.n_initial_queries = n_initial_queries
        self.seed = seed
        self.env_seed = seed
        self.max_queries = max_queries
        self.no_improvements_count = 0
        self.workerid = workerid
        self.recorder = recorder
        self.timestep = 0
        print("truth:")
        print(self.truth)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.hc.save_reward_model(self.env_name + "-reward-final")

    def train_reward_model(self, training_epochs):
        my_dataloader = self.prepare_training_data()
        self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)

    def train_reward_ensamble(self, ensamble, training_epochs):
        my_dataloader = self.prepare_training_data()
        for model in ensamble:
            meta_data = model.train_dataset_with_critical_points(my_dataloader, None, training_epochs)

    def prepare_training_data(self):
        o1, o2, prefs, critical_points = self.hc.get_all_preference_pairs_with_critical_points()
        tensor_o1 = torch.Tensor(o1)
        tensor_o2 = torch.Tensor(o2)
        tensor_prefs = torch.Tensor(prefs)
        tensor_critical_points = torch.Tensor(critical_points)
        my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs, tensor_critical_points)
        my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
        return my_dataloader

    @torch.no_grad()
    def collect_segments(self, model, test_episodes=5000, n_collect_segments=0, extra_env=None):
        """
        Collects segments from the environment.
        """
        total_segments = []
        critical_points = []

        env = gym.make(self.env_name)
        self.env_seed += 1
        env.reset(seed=self.env_seed)
        score = 0
        traj_segment = []
        segment_reward = 0
        n_critical_points = 0
        smallest_rew_index, largest_rew_index = -1, -1
        smallest_rew, largest_rew = self.smallest_rew_threshold, self.largest_rew_threshold
        running_average_rew = collections.deque(maxlen=3)
        latest_indexes = collections.deque(maxlen=3)
        for e in range(test_episodes):
            obs = reset_env(env)
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done = env_step(action, env)

                segment_reward += reward
                score += reward
                action = np.resize(action, (action.shape[0], ))

                running_average_rew.append(reward)
                running_average_rew_total = sum(running_average_rew) / len(running_average_rew)
                latest_indexes.append(len(traj_segment))

                if largest_rew < running_average_rew_total:
                    largest_rew_index = latest_indexes[np.argmax(running_average_rew)]
                    largest_rew = running_average_rew_total

                if smallest_rew > running_average_rew_total:
                    smallest_rew_index = latest_indexes[np.argmin(running_average_rew)]
                    smallest_rew = running_average_rew_total

                traj_segment.append(np.concatenate((obs.squeeze(), action)))

                if len(traj_segment) == self.traj_length:
                    self.process_traj_segment(traj_segment, segment_reward, done, self.traj_length)
                    total_segments.append([traj_segment, segment_reward])
                    traj_segment, segment_reward = [], 0

                    # Critical point extension
                    if smallest_rew < self.smallest_rew_threshold:
                        n_critical_points += 1
                    else:
                        smallest_rew_index = -1

                    if largest_rew > self.largest_rew_threshold:
                        n_critical_points += 1
                    else:
                        largest_rew_index = -1

                    self.hc.add_critical_points(smallest_rew_index, largest_rew_index)
                    critical_points.append([smallest_rew_index, largest_rew_index])

                    largest_rew = self.largest_rew_threshold
                    largest_rew_index = -1
                    smallest_rew = self.smallest_rew_threshold
                    smallest_rew_index = -1
                    running_average_rew.clear()
                    latest_indexes.clear()

                    if len(total_segments) % (n_collect_segments // 10) == 0:
                        print("Collected segments: " + str(len(total_segments)) + "/" + str(n_collect_segments))

                if len(total_segments) >= n_collect_segments != 0:
                    env.close()
                    return total_segments, critical_points
        env.close()
        return total_segments, critical_points

    def generate_segments(self, queries_multiplier=10, n_past_batches=5):
        """
        Generates new segments that can be used to form preference pairs.

        :param queries_multiplier: Determines how much to oversample realtive to the number of queries q * k
        :param n_past_batches: How many segment batches in the past we should consider
        """
        assert n_past_batches > 0
        traj_to_collect = self.n_queries * queries_multiplier
        self.collect_segments(self.model, 100000, traj_to_collect)
        start_index = max(0, (self.hc.segments_size - (traj_to_collect * n_past_batches)))
        end_index = self.hc.segments_size
        trajectories = self.hc.segments[start_index:end_index]
        critical_points = self.hc.critical_points[start_index:end_index]
        return trajectories, critical_points

    def process_traj_segment(self, traj_segment, segment_reward, done, traj_k_lenght=25):
        if len(traj_segment) < traj_k_lenght and done:
            while len(
                    traj_segment) < traj_k_lenght:
                traj_segment.append(traj_segment[-1])
        self.hc.add_segment(traj_segment, segment_reward)


    def update_params(self,
                      n_queries=10,
                      initial_reward_estimation_epochs=200,
                      reward_training_epochs=50,
                      traj_length=50,
                      n_initial_queries=200,
                      max_queries=1400,
                      ):
        self.n_queries = n_queries
        self.initial_reward_estimation_epochs = initial_reward_estimation_epochs
        self.reward_training_epochs = reward_training_epochs
        self.traj_length = traj_length
        self.n_initial_queries = n_initial_queries
        self.max_queries = max_queries


class UpdateRewardFunctionCriticalPoint(PreferenceCallback):
    """
    Callback to periodically gather query feedback and updates the reward model.
    """
    def __init__(self, **kwargs):
        super(UpdateRewardFunctionCriticalPoint, self).__init__(**kwargs)
        print("UpdateRewardFunctionCriticalPoint")

    def _on_training_start(self) -> None:
        print("Performing initial training of reward model")
        self.hc.writer = SummaryWriter(self.logger.get_dir())

        # Collect segments
        trajectories, critical_points = self.generate_segments(queries_multiplier=5, n_past_batches=1)

        # Generate prefernce pairs (Using random sampling)
        self.hc.generate_preference_pairs_uniform(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_initial_queries)

        # Train the reward model
        self.train_reward_model(self.initial_reward_estimation_epochs)

    def _on_step(self) -> bool:

        if self.hc.pairs_size < self.max_queries:
            # Collect segments
            trajectories, critical_points = self.generate_segments(queries_multiplier=5)

            # Generate prefernce pairs (Using random sampling)
            self.hc.generate_preference_pairs_uniform(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_queries)

            # Train the reward model
            self.train_reward_model(self.reward_training_epochs)
        return True


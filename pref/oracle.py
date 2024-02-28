import numpy as np
import torch
import wandb
from numpy.random import choice
from torch import nn

import random
from random import sample
from pref.utils import save_pickle, load_pickle


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        if j < len(sizes) - 2:
            layers += [nn.Dropout(0.5 if j > 0 else 0.2)]
    return nn.Sequential(*layers)


class HumanRewardNetwork(nn.Module):
    def __init__(self, obs_size, hidden_sizes=(64, 64)):
        super(HumanRewardNetwork, self).__init__()
        self.linear_relu_stack = mlp([obs_size] + list(hidden_sizes) + [1], activation=nn.LeakyReLU)
        self.tanh = nn.Tanh()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.tanh(logits)


class HumanCritic:
    LEARNING_RATE = 0.0003
    BUFFER_SIZE = 1e5
    BATCH_SIZE = 10

    def __init__(self,
                 obs_size=3,
                 action_size=2,
                 maximum_segment_buffer=1000000,
                 maximum_preference_buffer=3500,
                 training_epochs=10,
                 batch_size=32,
                 hidden_sizes=(64, 64),
                 traj_k_lenght=100,
                 weight_decay=0.0,
                 learning_rate=0.0003,
                 regularize=False,
                 env_name=None,
                 custom_oracle=False,
                 seed=12345,
                 epsilon=0.1):
        print("created")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ===BUFFER===
        self.segments = [None] * maximum_segment_buffer  # lists are very fast for random access
        self.pairs = [None] * maximum_preference_buffer
        self.critical_points = [None] * maximum_segment_buffer
        self.maximum_segment_buffer, self.maximum_preference_buffer, self.maximum_critical_points_buffer = maximum_segment_buffer, maximum_preference_buffer, maximum_segment_buffer
        self.segments_index, self.pairs_index, self.critical_points_index = 0, 0, 0
        self.segments_size, self.pairs_size, self.critical_points_size = 0, 0, 0
        self.segments_max_k_len = traj_k_lenght

        # === MODEL ===
        self.obs_size = obs_size
        self.action_size = action_size
        self.SIZES = hidden_sizes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.init_model()  # creates model
        self.fish = 0

        # === DATASET TRAINING ===
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        self.writer = None  # SummaryWriter(working_path + reward_model_name)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.updates = 0

        self.pos_discount_start_multiplier = 1.0
        self.pos_discount = 0.7
        self.min_pos_discount = 0.01

        self.neg_discount_start_multiplier = 1.0
        self.neg_discount = 0.7
        self.min_neg_discount = 0.01

        self.regularize = regularize
        self.custom_oracle = custom_oracle
        self.env_name = env_name
        self.oracle_reward_function = self.get_oracle_reward_function(env_name)
        self.epsilon = epsilon

        if wandb.run is not None:
            wandb.define_metric("oracle/pairs")
            wandb.define_metric("oracle/*", step_metric="oracle/pairs")
            wandb.define_metric("oracle/total", summary="mean")
            wandb.define_metric("oracle/true_rew", summary="mean")

            wandb.define_metric("reward/updates")
            wandb.define_metric("reward/*", step_metric="reward/updates")

    def update_params(self,
                      maximum_segment_buffer=1000000,
                      maximum_preference_buffer=3500,
                      batch_size=32,
                      hidden_sizes=(64, 64),
                      traj_k_lenght=100,
                      learning_rate=0.0003,
                      weight_decay=0.00001):

        # ===BUFFER===
        self.segments = [None] * maximum_segment_buffer  # lists are very fast for random access
        self.pairs = [None] * maximum_preference_buffer
        self.critical_points = [None] * maximum_segment_buffer
        self.maximum_segment_buffer, self.maximum_preference_buffer, self.maximum_critical_points_buffer = maximum_segment_buffer, maximum_preference_buffer, maximum_segment_buffer
        self.segments_index, self.pairs_index, self.critical_points_index = 0, 0, 0
        self.segments_size, self.pairs_size, self.critical_points_size = 0, 0, 0
        self.segments_max_k_len = traj_k_lenght

        self.SIZES = hidden_sizes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.init_model()

        self.batch_size = batch_size

        self.updates = 0
    def get_oracle_reward_function(self, env_name):
        return self.get_query_results_reward

    def init_model(self, delete=False):
        # ==MODEL==
        if delete:
            del self.reward_model
            del self.optimizer
        self.reward_model = HumanRewardNetwork(self.obs_size[0] + self.action_size, self.SIZES)

        # ==OPTIMIZER==
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def clear_segment(self):
        self.segments = [None] * self.maximum_segment_buffer
        self.segments_size, self.segments_index = 0, 0

    def add_segment(self, o, total_reward):
        assert len(o) <= self.segments_max_k_len
        self.segments[self.segments_index] = [o, total_reward]
        self.segments_size = min(self.segments_size + 1, self.maximum_segment_buffer)
        self.segments_index = (self.segments_index + 1) % self.maximum_segment_buffer

    def add_critical_points(self, min_index, max_index):
        self.critical_points[self.critical_points_index] = [min_index, max_index]
        self.critical_points_size = min(self.critical_points_size + 1, self.maximum_critical_points_buffer)
        self.critical_points_index = (self.critical_points_index + 1) % self.maximum_critical_points_buffer

    def add_pairs(self, o0, o1, preference):
        self.pairs[self.pairs_index] = [o0, o1, preference]
        self.pairs_size = min(self.pairs_size + 1, self.maximum_preference_buffer)
        self.pairs_index = (self.pairs_index + 1) % self.maximum_preference_buffer

    def add_pairs_with_critical_points(self, o0, o1, preference, critical_points):
        self.pairs[self.pairs_index] = [o0, o1, preference, critical_points]
        self.pairs_size = min(self.pairs_size + 1, self.maximum_preference_buffer)
        self.pairs_index = (self.pairs_index + 1) % self.maximum_preference_buffer

    def random_sample_segments(self, number_of_sampled_segments=64):
        idxs = sample(range(self.segments_size), number_of_sampled_segments)
        return [self.segments[idx] for idx in idxs]

    def save_buffers(self, path="", env_name="", save_name="buffer"):
        save_pickle(self.segments, path + "segments_" + env_name + save_name)
        save_pickle(self.segments_size, path + "segments_size_" + env_name + save_name)
        save_pickle(self.segments_index, path + "segments_index_" + env_name + save_name)
        save_pickle(self.pairs, path + "pairs_" + env_name + save_name)
        save_pickle(self.pairs_size, path + "pairs_size_" + env_name + save_name)
        save_pickle(self.pairs_index, path + "pairs_index_" + env_name + save_name)
        save_pickle(self.critical_points, path + "critical_points_" + env_name + save_name)
        save_pickle(self.critical_points_size, path + "critical_points_size_" + env_name + save_name)
        save_pickle(self.critical_points_index, path + "critical_points_index_" + env_name + save_name)

    def load_buffers(self, path="", env_name="", load_name="buffer"):
        self.segments = load_pickle(path + "segments_" + env_name + load_name)
        self.segments_size = load_pickle(path + "segments_size_" + env_name + load_name)
        self.segments_index = load_pickle(path + "segments_index_" + env_name + load_name)
        self.pairs = load_pickle(path + "pairs_" + env_name + load_name)
        self.pairs_size = load_pickle(path + "pairs_size_" + env_name + load_name)
        self.pairs_index = load_pickle(path + "pairs_index_" + env_name + load_name)
        self.pairs = load_pickle(path + "critical_points_" + env_name + load_name)
        self.pairs_size = load_pickle(path + "critical_points_size_" + env_name + load_name)
        self.pairs_index = load_pickle(path + "critical_points_index_" + env_name + load_name)


    def train_dataset_with_critical_points(self, dataset, meta_data, epochs_override=-1):
        max_regularization_sum = 0
        reg_sum = 1
        for i in range(5):
            max_regularization_sum += reg_sum
            reg_sum = reg_sum * 0.7

        epochs = epochs_override if epochs_override != -1 else self.training_epochs

        self.reward_model.train(True)
        avg_loss = 0
        meta_data = {}
        episode_loss = 0

        for epoch in range(1, epochs + 1):

            running_loss = 0
            running_accuracy = 0

            for step, (o1, o2, prefs, critical_points) in enumerate(dataset):
                self.optimizer.zero_grad()
                # TODO this is RNN code
                o1_unrolled = torch.reshape(o1, [-1, self.obs_size[0] + self.action_size])
                o2_unrolled = torch.reshape(o2, [-1, self.obs_size[0] + self.action_size])
                r1_unrolled = self.reward_model(o1_unrolled)
                r2_unrolled = self.reward_model(o2_unrolled)

                r1_rolled = torch.reshape(r1_unrolled, o1.shape[0:2])
                r2_rolled = torch.reshape(r2_unrolled, o2.shape[0:2])

                rs1 = torch.sum(r1_rolled, dim=1)
                rs2 = torch.sum(r2_rolled, dim=1)
                rss = torch.stack([rs1, rs2])
                rss = torch.t(rss)

                preds = torch.softmax(rss, dim=0)
                preds_correct = torch.eq(torch.argmax(prefs, 1), torch.argmax(preds, 1)).type(torch.float32)
                accuracy = torch.mean(preds_correct)

                approve_reward, punishment_reward, n_approve, n_punishment = self.get_critical_points_rewards(
                    critical_points, prefs, r1_rolled,
                    r2_rolled)
                n_approve = n_approve if n_approve != 0 else 1
                n_punishment = n_punishment if n_punishment != 0 else 1

                approve_reward = approve_reward / n_approve
                punishment_reward = punishment_reward / n_punishment

                if self.regularize:
                    regularization_approve = abs(max_regularization_sum - approve_reward)
                    regularization_punishment = abs(max_regularization_sum + punishment_reward)

                    loss_pref = -torch.sum(torch.log(preds[prefs == 1]))
                    if self.env_name == "Walker2d-v3":
                        loss = loss_pref - approve_reward * 10 + punishment_reward * 5
                    elif self.env_name == "HalfCheetah-v3":
                        loss = loss_pref - approve_reward * 10 + punishment_reward * 10
                    else:
                        loss = loss_pref - approve_reward * 10 + punishment_reward * 5
                else:
                    loss_pref = -torch.sum(torch.log(preds[prefs == 1]))
                    loss = loss_pref

                running_loss += loss.detach().numpy().item()
                running_accuracy += accuracy


                reporting_interval = (self.training_epochs // 10) if self.training_epochs >= 10 else 1
                if epoch % reporting_interval == 0 and step == len(dataset) - 1:
                    print("Epoch %d , Training loss (for one batch) at step %d: %.4f, Accuracy %.4f" % (epoch, step, float(loss), float(accuracy)))
                    print("Seen so far: %s samples" % ((step + 1) * self.batch_size))

                loss.backward()
                self.optimizer.step()

            episode_loss = (running_loss / len(dataset))
            avg_loss += episode_loss
            episode_accuracy = (running_accuracy / len(dataset))
            if self.writer:
                self.writer.add_scalar("reward/loss", episode_loss, self.updates)
                self.writer.add_scalar("reward/accuracy", episode_accuracy, self.updates)
            if wandb.run is not None:
                wandb.log({"reward/loss": episode_loss,
                           "reward/accuracy": episode_accuracy,
                           "reward/updates": self.updates
                           })
            self.updates += 1

        avg_loss = avg_loss / epochs
        if (avg_loss - episode_loss) < 2.5:
            print("episode_loss:" + str(episode_loss))
            print("avg_loss:" + str(avg_loss))
            meta_data['improved'] = False
        else:
            meta_data['improved'] = True
        self.reward_model.train(False)
        return meta_data

    def get_critical_points_rewards(self, critical_points, prefs, r1_rolled, r2_rolled):
        critical_points_discounted_reward_punishment = torch.zeros_like(r1_rolled)
        critical_points_discounted_reward_approve = torch.zeros_like(r1_rolled)
        for i in range(len(prefs)):
            if prefs[i][0] == 1:
                critical_points_discounted_reward_punishment[i] = r1_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r1_rolled[i] * critical_points[i, :, 1]
            if prefs[i][1] == 1:
                critical_points_discounted_reward_punishment[i] = r2_rolled[i] * critical_points[i, :, 0]
                critical_points_discounted_reward_approve[i] = r2_rolled[i] * critical_points[i, :, 1]

        punishments_in_batch = torch.sum(critical_points[:, :, 0] == 1).item()
        approvements_in_batch = torch.sum(critical_points[:, :, 1] == 1).item()

        punishment_reward = torch.sum(critical_points_discounted_reward_punishment) #/ punishments_in_batch
        approve_reward = torch.sum(critical_points_discounted_reward_approve) #/ approvements_in_batch
        return approve_reward, punishment_reward, approvements_in_batch, punishments_in_batch

    def generate_data_for_training_with_critical_points(self, queries):
        queries = np.array(queries, dtype=object)
        o1, o2, prefs, critical_points = queries[:, 0, 0], queries[:, 1, 0], queries[:, 2], queries[:, 3]
        o1 = [np.stack(segments) for segments in o1]
        o2 = [np.stack(segments) for segments in o2]

        critical_points = self.generate_critical_point_segment(critical_points)
        prefs = np.asarray(prefs).astype('float32')
        return o1, o2, prefs, critical_points


    def generate_critical_point_segment(self, critical_points):
        rolled_critical_points = [[[0, 0] for _ in range(self.segments_max_k_len)] for _ in range(len(critical_points))]
        for i in range(len(critical_points)):
            neg_index = critical_points[i][0]
            pos_index = critical_points[i][1]

            if pos_index != -1:
                current_pos_discount = self.pos_discount_start_multiplier
                for j in reversed(range(max(0, pos_index - 5), pos_index + 1)):
                    rolled_critical_points[i][j][1] = max(current_pos_discount, self.min_pos_discount)
                    current_pos_discount *= self.pos_discount

            if neg_index != -1:
                current_neg_discount = self.neg_discount_start_multiplier
                for j in reversed(range(max(0, neg_index - 5), neg_index + 1)):
                    rolled_critical_points[i][j][0] = max(current_neg_discount, self.min_neg_discount)
                    current_neg_discount *= self.neg_discount
        critical_points = np.asarray(rolled_critical_points).astype('float32')
        return critical_points

    def save_reward_model(self, env_name="LunarLanderContinuous-v2"):
        torch.save(self.reward_model.state_dict(), env_name)

    def load_reward_model(self, env_name="LunarLanderContinuous-v2"):
        print("loading:" + "models/reward_model/" + env_name)
        self.reward_model.load_state_dict(torch.load(env_name))


    def get_all_preference_pairs_human(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs = self.generate_data_for_training_human(pairs)
        return obs1, obs2, prefs

    def get_all_preference_pairs_with_critical_points(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs, critical_points = self.generate_data_for_training_with_critical_points(pairs)
        return obs1, obs2, prefs, critical_points

    def generate_preference_pairs_uniform(self, trajectories, critical_points, number_of_queries=200, truth=100):
        for _ in range(number_of_queries):
            segments, points = self.random_sample_batch_segments_with_critical_points(trajectories, critical_points, number_of_sampled_segments=2)
            query = self.oracle_reward_function(segments[0], segments[1], truth, points)
            self.add_pairs_with_critical_points(query[0], query[1], query[2], query[3])

    def random_sample_batch_segments(self, trajectories, number_of_sampled_segments=64):
        idxs = sample(range(len(trajectories)), number_of_sampled_segments)
        return [trajectories[idx] for idx in idxs]

    def random_sample_batch_segments_with_critical_points(self, trajectories, critical_points, number_of_sampled_segments=64):
        idxs = sample(range(len(trajectories)), number_of_sampled_segments)
        return [trajectories[idx] for idx in idxs], [critical_points[idx] for idx in idxs]

    def get_query_results_reward(self, segment1, segment2, truth, critical_points):
        total_reward_1 = segment1[-1]
        total_reward_2 = segment2[-1]
        truth_percentage = truth / 100.0
        fakes_percentage = 1 - truth_percentage
        epsilon = self.epsilon
        if total_reward_1 > total_reward_2 + epsilon:
            preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
            point = critical_points[0] if preference[0] == 1 else critical_points[1]

        elif total_reward_1 + epsilon < total_reward_2:
            preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
            point = critical_points[1] if preference[1] == 1 else critical_points[0]
        else:
            preference = [0, 0]
            point = [-1, -1]
        return [segment1, segment2, preference, point]


    def predict_reward(self, obs):
        return self.reward_model(obs)
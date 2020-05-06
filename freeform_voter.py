# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/stable-baselines/')
import glob

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import tensorflow as tf
from collections import defaultdict, deque
import pandas as pd

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams.update({'font.size': 14})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cloudpickle

from tqdm import tqdm
import random
import fire
import numpy as np
import gym
import gym.spaces
import copy
import os
import pickle
import json
import bz2
import gzip
import pickletools
import freeform_trolley


class PreferenceEnv:
    def __init__(self, n_agents, n_actions, n_steps, know_other_preferences, stochastic_voting, cost_exponent, mean_of_std, std_of_mean):
        # TODO: handle credence (if necessary)
        self.num_agents = n_agents
        self.num_actions = n_actions
        self.max_steps = n_steps
        self.action_space = gym.spaces.Box(0 if stochastic_voting else -np.inf, np.inf, (n_actions,), np.float32)
        # self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((n_actions + 1) * (n_agents if know_other_preferences else 1) + 1,), np.float32)
        self.metadata = {}
        self.know_other_preferences = know_other_preferences
        self.stochastic_voting = stochastic_voting
        self.cost_exponent = cost_exponent
        self.mean_of_std = mean_of_std
        self.std_of_mean = std_of_mean
        self.reset()

    def seed(self, n):
        pass

    def _generate_preferences(self):
        self.preferences = []
        for _ in range(self.num_agents):
            mean = np.random.randn() * self.std_of_mean
            std = np.random.exponential(self.mean_of_std)
            self.preferences.append(np.random.randn(self.num_actions) * std + mean)
        self.preferences = np.array(self.preferences)

    def _get_state(self):
        state = []
        for p, b in zip(self.preferences, self.remaining_budgets):
            state.append(list(p) + [b] + ([] if self.know_other_preferences else[self.remaining_steps]))
        if self.know_other_preferences:
            state = [sum(state, []) + [self.remaining_steps]] * self.num_agents
        return np.array(state)

    def step(self, orig_actions):
#        actions = []
#        for i in range(self.num_agents):
#            actions.append((self.preferences[i] + transforms[i][0]) * transforms[i][1])

        actions = copy.deepcopy(orig_actions)
        for i in range(self.num_agents):
            cost_paid = np.sum(np.abs(actions[i])**self.cost_exponent)
            if cost_paid > self.remaining_budgets[i]:
                actions[i] *= (self.remaining_budgets[i] / cost_paid)**(1 / self.cost_exponent)
                cost_paid = np.sum(np.abs(actions[i])**self.cost_exponent)
                assert abs(cost_paid - self.remaining_budgets[i]) < 0.0001
            self.remaining_budgets[i] = max(self.remaining_budgets[i] - cost_paid, 0)
        votes = np.sum(actions, axis=0)

        if self.stochastic_voting:
            votes += 0.000001
            votes /= np.sum(votes)
            chosen = np.random.choice(list(range(self.num_actions)), p=votes)
        else:
            # TODO: handle random tie-breaking (?)
            chosen = np.argmax(votes)

        rewards = self.preferences[:, chosen]
        self._generate_preferences()
        self.remaining_steps -= 1
        return self._get_state(), rewards, self.remaining_steps <= 0, {}

    def reset(self, *args, **kwargs):
        self.remaining_steps = self.max_steps
        self.remaining_budgets = [self.max_steps * self.num_actions] * self.num_agents
        self._generate_preferences()
        return self._get_state()


def standardize(v):
    return (v - np.mean(v, axis=1, keepdims=True)) / np.std(v, axis=1, keepdims=True)


def get_n_on_tracks_fct(distribution, argument, continuous):
    if distribution == 'single':
        return lambda: argument
    if distribution == 'exp':
        assert continuous
        return lambda: np.random.exponential(argument)
    if distribution == 'oneto':
        if continuous:
            return lambda: np.random.rand() * (argument - 1) + 1
        else:
            return lambda: np.random.randint(1, argument + 1)
    assert False, f'Unknown distribution: {distribution}'


def possible_values_dist(distribution, argument):
    if distribution == 'single':
        return [argument]
    if distribution == 'oneto':
        return list(range(1, argument + 1))
    assert False


def process_rewards(theory, rewards):
    total = 0
    for k in theory:
        found_at_least_one = False
        for k2 in rewards:
            if k in k2:
                total += theory[k] * rewards[k2]
                found_at_least_one = True
        assert found_at_least_one
    return total


class NashEnv:
    def __init__(self, theories, get_credences, env, stochastic_voting, cost_exponent, rand_adv, is_testing):
        self.is_testing = is_testing
        self.all_theories = theories
        if rand_adv:
            self.theories = [None, None]
        else:
            self.theories = theories
        self.get_credences = get_credences
        self.env = env
        self.num_agents = len(self.theories)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (env.action_space.n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (env.observation_space.shape[0] + 1 + len(self.theories) + rand_adv,), np.float32)
        self.metadata = {}
        self.stochastic_voting = stochastic_voting
        self.cost_exponent = cost_exponent
        self.rand_adv = rand_adv
        self.cur_steps = 0
        self.recent_steps = deque(maxlen=100)
        self.default_budget = 10.0
        self.reset()

    def reset(self, credences=None, number_on_tracks=None):
        self.extra_obs = [[]] * len(self.theories)
        if self.rand_adv:
            idxs = [0, 1]
            if not self.is_testing:
                idxs = np.random.choice(list(range(len(self.all_theories))), 2)
            self.theories = [self.all_theories[idxs[0]], self.all_theories[idxs[1]]]
            self.extra_obs = [[idxs[0]], [idxs[1]]]
        obs = list(self.env.reset(number_on_tracks))
        self.credences = credences if credences is not None else np.array(self.get_credences())
        self.remaining_budgets = [self.default_budget] * self.num_agents
        self.cur_steps = 0
        return np.array([list(obs) + [self.remaining_budgets[i]] + list(self.credences) + self.extra_obs[i] for i in range(len(self.theories))])

    def step(self, orig_actions, verbose=False):
        actions = copy.deepcopy(orig_actions)
        if self.stochastic_voting:
            actions = np.exp(actions)
        for i in range(self.num_agents):
            cost_paid = np.sum(np.abs(actions[i])**self.cost_exponent)
            if verbose:
                print('budget', cost_paid, self.remaining_budgets)
            if cost_paid > self.remaining_budgets[i]:
                actions[i] *= (self.remaining_budgets[i] / cost_paid)**(1 / self.cost_exponent)
                cost_paid = np.sum(np.abs(actions[i])**self.cost_exponent)
                assert abs(cost_paid - self.remaining_budgets[i]) < 0.0001
            self.remaining_budgets[i] = max(self.remaining_budgets[i] - cost_paid, 0)
        if verbose:
            print(actions)
            import ipdb; ipdb.set_trace()
        votes = np.sum(actions * self.credences[:, None], axis=0)
        if verbose:
            print(votes)

        if self.stochastic_voting:
            votes += 0.000001
            votes /= np.sum(votes)
            chosen = np.random.choice(list(range(self.action_space.n)), p=votes)
        else:
            # TODO: handle random tie-breaking (?)
            chosen = np.argmax(votes)

        obs, rewards, done, info = self.env.step(chosen)

        self.cur_steps += 1
        if done:
            self.recent_steps.append(self.cur_steps)

        return (
            np.array([list(obs) + [self.remaining_budgets[i]] + list(self.credences) + self.extra_obs[i] for i in range(len(self.theories))]),
            np.array([process_rewards(theory, rewards) for theory in self.theories]),
            done,
            mergedict(rewards, info)
        )

    def seed(self, n):
        pass


def mergedict(a, b):
    d = {}
    for k in a:
        d[k] = a[k]
    for k in b:
        assert k not in d
        d[k] = b[k]
    return d


class TabularSarsa:
    def __init__(self, n_actions, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.table = {} # defaultdict(lambda: np.zeros(n_actions))
        # Convention: the None state corresponds to done=True
        self.table[None] = np.zeros(n_actions)

    def _get_table(self, state):
        if state not in self.table:
            self.table[state] = np.zeros(self.n_actions)
        return self.table[state]

    def predict(self, states, force_inside=False):
        res = []
        for state in states:
            state = tuple(state)
            if force_inside:
                if state not in self.table:
                    print(state)
                    print(list(self.table.keys())[:2])
                assert state in self.table
            res.append(self._get_table(state))
        return res

    def learn(self, states, actions, rewards, next_states, next_actions, dones):
        # Q(s, a) = lr*(r + gamma Q(s', a')) + (1-lr)*Q(s, a)
        for s, a, r, sp, ap, d in zip(states, actions, rewards, next_states, next_actions, dones):
            assert s is not None
            self.table[s][a] = (
                    (1 - self.learning_rate) * self._get_table(s)[a] +
                    self.learning_rate * (r + self.gamma * (self._get_table(sp)[ap] if not d else 0))
            )

    def save_data(self):
        return self.table

    def load_data(self, table):
        self.table = table


class SarsaModel(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DeepSarsa:
    def __init__(self, n_inputs, n_actions, learning_rate, gamma, min_batch_size, is_deepq):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = SarsaModel(n_inputs, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.min_batch_size = min_batch_size
        self.is_deepq = is_deepq
        self._init_batch()

    def _init_batch(self):
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_next_states = []
        self.batch_next_actions = []
        self.batch_dones = []

    def predict(self, states, force_inside=False):
        return self.model(torch.Tensor(states)).detach().numpy()

    def learn(self, states, actions, rewards, next_states, next_actions, dones):
        self.batch_states += states
        self.batch_actions += actions
        self.batch_rewards += rewards
        self.batch_next_states += next_states
        self.batch_next_actions += next_actions
        self.batch_dones += dones
        if len(self.batch_states) > self.min_batch_size:
            # Q(s, a) = lr*(r + gamma Q(s', a')) + (1-lr)*Q(s, a)
            self.optimizer.zero_grad()
            if self.is_deepq:
                target_next = self.model(torch.Tensor(self.batch_next_states)).detach().max(axis=1)[0]
            else:
                target_next = self.model(torch.Tensor(self.batch_next_states)).detach()[
                    list(range(len(self.batch_next_states))), self.batch_next_actions]

            targets = (
                    torch.Tensor(self.batch_rewards) +
                    (
                        self.gamma *
                        target_next *
                        torch.Tensor(1 - np.array(self.batch_dones))
                    )
            )
            sources = self.model(torch.Tensor(self.batch_states))[list(range(len(self.batch_states))), self.batch_actions]
            # if random.random() < 0.01:
            #     import ipdb; ipdb.set_trace()
            loss = F.mse_loss(sources, targets)
            loss.backward()
            self.optimizer.step()
            self._init_batch()

    def save_data(self):
        return self.model.state_dict()

    def load_data(self, state_dict):
        self.model.load_state_dict(state_dict)


class RollingMeanOfStd:
    def __init__(self, max_n=None):
        self.n = 0
        self.sum = 0
        self.max_n = max_n
        self.rolling = deque()
        # TODO: make this optional
        self.add(1)

    def mean_std(self, default=None):
        if self.n <= 0:
            if default is not None:
                return default
            else:
                assert False, 'Requesting a mean with 0 samples'
        return np.sqrt(self.sum / self.n)

    def add(self, v):
        self.sum += v**2
        self.n += 1
        if self.max_n is not None:
            self.rolling.append(v**2)
            while self.n > self.max_n:
                self.sum -= self.rolling.popleft()
                self.n -= 1


class VarianceNet(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.exp(self.fc3(x))


class LearnedVariance:
    def __init__(self, n_credences, batch_size, learning_rate):
        self.net = VarianceNet(n_credences)
        self.batch_size = batch_size
        self.batch_x = []
        self.batch_y = []
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def add(self, credences, v):
        self.batch_x.append(credences)
        self.batch_y.append([v**2])
        if len(self.batch_x) > self.batch_size:
            self.optimizer.zero_grad()
            values = self.net(torch.Tensor(self.batch_x))
            loss = F.mse_loss(values, torch.Tensor(self.batch_y))
            loss.backward()
            self.optimizer.step()

            self.batch_x = []
            self.batch_y = []

    def mean_std(self, credences):
        res = self.net(torch.Tensor(credences))
        return np.sqrt(res.detach().numpy()[0])

    def save_data(self):
        return self.net.state_dict()

    def load_data(self, data):
        self.net.load_state_dict(data)


class TabularVariance:
    def __init__(self, rolling_window):
        self.var = defaultdict(lambda: RollingMeanOfStd(max_n=rolling_window))

    def add(self, credence, v):
        self.var[credence].add(v)

    def mean_std(self, credences):
        return self.var[credences].mean_std()

    def save_data(self):
        return dict(self.var)

    def load_data(self, var):
        for k, v in var:
            self.var[k] = v

class VarianceModel:
    def __init__(self, theories, get_credences, env, get_epsilon, model_type, credence_round,
                 n_track_adjust, learn_with_explore, lr, rolling_window, batch_size, variance_type, do_variance,
                 stochastic):
        self.theories = theories
        self.get_credences = get_credences
        self.env = env
        self.do_variance = do_variance
        self.learn_with_explore = learn_with_explore
        self.credence_round = credence_round
        self.stochastic = stochastic
        if model_type == 'tabular':
            self.models = [TabularSarsa(env.action_space.n, lr, 1.0) for _ in theories]
        elif 'deep' in model_type:
            self.models = [DeepSarsa(env.observation_space.shape[0] + len(theories), env.action_space.n, lr, 1.0, batch_size, model_type=='deepq') for _ in theories]
        else:
            assert False
        self.n_track_adjust = n_track_adjust
        if variance_type == 'tabular':
            self.variances = [TabularVariance(rolling_window) for _ in theories]#defaultdict(lambda: [RollingMeanOfStd(max_n=rolling_window) for _ in theories])
        else:
            self.variances = [LearnedVariance(len(get_credences()), batch_size, lr) for _ in theories]
        self.obs = None
        self.get_epsilon = get_epsilon
        self.num_timesteps = 0

    def reset(self, credences=None, number_on_tracks=None):
        self.credences = np.array(credences if credences is not None else self.get_credences())
        self.raw_obs = self.env.reset(number_on_tracks)
        return self._get_state()

    def _get_state(self):
        if self.raw_obs is not None:
            return tuple(list(self.raw_obs[:-1]) + [self.n_track_adjust(self.raw_obs[-1])] + list(self.credence_round(self.credences)))
        return None

    def step(self, action):
        self.raw_obs, reward, done, info = self.env.step(action)
        rewards = []
        for t in self.theories:
            rewards.append(process_rewards(t, reward))
        return self._get_state(), rewards, done, mergedict(reward, info)

    def predict(self, obs, add=False, deterministic=False, verbose=False):
        action_scores = np.array([model.predict([obs], deterministic)[0] for model in self.models])
        if add:
            for std, a in zip(self.variances, action_scores):
                std.add(tuple(self.credences), np.std(a))

        stds = np.array([v.mean_std(tuple(self.credences)) for v in self.variances])
        if self.do_variance:
            normalized_scores = (action_scores - np.mean(action_scores, axis=1)[:, None]) / (stds[:, None] + 0.000001)
        else:
            normalized_scores = action_scores
        if self.stochastic:
            normalized_scores -= np.min(normalized_scores, axis=1)[:, None]
        votes = np.sum(normalized_scores * self.credences[:, None], axis=0)
        if self.stochastic:
            chosen = np.random.choice(list(range(len(votes))), p=votes)
        else:
            chosen = np.argmax(votes)
        return chosen, None

    def learn(self, total_timesteps, callback=None):
        obs = self.reset()
        rewards = None
        prev_obs = None
        prev_a = None
        done = False

        since_done = 0

        for i in tqdm(range(total_timesteps)):
            chosen, _ = self.predict(obs, add=True)
            epsilon = self.get_epsilon(i)
            action = chosen
            if random.random() < epsilon:
                # NOTE: This could be changed somehow to do off-policy learning (?)
                # The way to do this would be to save chosen somewhere else and use it ONLY as the second SARSA
                # action.
                action = self.env.action_space.sample()
                if self.learn_with_explore:
                    # If learn_with_explore is True, sarsa takes into account the exploration policy,
                    # otherwise it does not
                    chosen = action

            if prev_obs is not None:
                for model, theory, reward in zip(self.models, self.theories, rewards):
                    model.learn([prev_obs], [prev_a], [reward], [obs], [chosen], [done])

            prev_obs = obs
            prev_a = action

            obs, rewards, done, _ = self.step(action)
            since_done += 1
            if done:
                # print(since_done)
                since_done = 0
                obs = self.reset()
            self.num_timesteps += 1
            if callback is not None:
                callback(locals(), globals())

            if i % 20000 == 0 and hasattr(self.models[0], 'table'):
                tqdm.write(f'{len(self.models[0].table)}')

    def save(self, path):
        data = ([v.save_data() for v in self.variances], [model.save_data() for model in self.models])
        gzip.open(path, 'wb').write(pickletools.optimize(pickle.dumps(data)))

    def load(self, path):
        variances, model_datas = pickle.load(gzip.open(path, 'rb'))
        for v, data in zip(self.variances, variances):
            v.load_data(data)
        for model, data in zip(self.models, model_datas):
            model.load_data(data)
        return self


class SequentialEnv:
    '''An environment that plays the subenvironment multiple times in a row before finally returning done.'''
    def __init__(self, env, n_sequence):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (env.observation_space.shape[0] + 1,), np.float32)
        self.n_sequence = n_sequence

    def reset(self, *args, **kwargs):
        self.remaining = self.n_sequence
        return [self.remaining] + list(self.env.reset(*args, **kwargs))

    def step(self, *args, **kwargs):
        s, r, d = self.env.step(*args, **kwargs)
        info = {'subenv_done': d}
        if d:
            self.remaining -= 1
            if self.remaining > 0:
                d = False
                # TODO: what to do about the environment args and kwargs here?
                s = self.env.reset()
        return [self.remaining] + list(s), r, d, info


class LRHalver:
    def __init__(self, start_lr, n_halves):
        self.lr = start_lr
        self.n_halves = n_halves
        self.prev_progress = 0

    def __call__(self, progress):
        progress = 1 - progress
        if int(progress * self.n_halves) != int(self.prev_progress * self.n_halves):
            print('Halving lr at', progress)
            self.lr /= 2
        self.prev_progress = progress
        return self.lr


class FreeformVoter:
    def __init__(self):
        self.n_calls = 0
        self.timesteps_so_far = 0

    def _get_trolley_model(self, is_testing):
        if self.env_args['credences'] is not None:
            assert len(self.env_args['theories']) == len(self.env_args['credences'])
            assert np.abs(np.sum(self.env_args['credences']) - 1) < 0.001, 'Credences do not sum to 1'
            credences = self.env_args['credences'] / np.sum(self.env_args['credences'])
        else:
            def _get_cred():
                # arr = np.array(np.random.rand(len(theories)))
                # probs = arr / arr.sum()
                a = np.random.rand()
                probs = np.array([a, 1 - a])
                if self.env_args['variance_type'] == 'tabular' or self.env_args['sarsa_type'] == 'tabular':
                    probs = np.round(probs * self.env_args['credence_granularity']) / self.env_args['credence_granularity']
                return probs
            credences = _get_cred
        trolley = lambda: SequentialEnv(
            freeform_trolley.TrolleyEnv(
                level=self.env_args['level'],
                number_on_tracks_fn=get_n_on_tracks_fct(
                    self.env_args['on_track_dist'], self.env_args['on_track'], continuous=(self.env_args['sarsa_type']!='tabular')
                )
            ),
            self.env_args['n_sequential']
        )
        if self.env_args['voting'] == 'nash':
            env_creator = lambda: NashEnv(self.env_args['theories'], credences, trolley(),
                                          stochastic_voting=self.env_args['stochastic_voting'],
                                          cost_exponent=self.env_args['cost_exponent'],
                                          rand_adv=self.env_args.get('rand_adv', False),
                                          is_testing=is_testing)
            env = DummyVecEnv([env_creator] * self.env_args['nenvs'])

            model = PPO2("MlpPolicy", env, verbose=1,
                         seed=self.env_args['seed'] if self.env_args['seed'] > 0 else None, gamma=1.0,
                         ent_coef=0.03,
                         learning_rate=self.env_args['learning_rate'])#(self.env_args['learning_rate'], self.env_args['n_halves']))
        elif self.env_args['voting'] == 'variance' or self.env_args['voting'] == 'mec':
            # assert self.env_args['cost_exponent'] == 2
            if self.env_args['sarsa_type'] == 'tabular':
                credence_round = lambda credences: np.round(credences * self.env_args['credence_granularity']).astype(np.int32)
            else:
                credence_round = lambda x: x
            model = VarianceModel(
                theories=self.env_args['theories'], get_credences=credences, env=trolley(),
                get_epsilon=lambda i: self.env_args['sarsa_eps'], model_type=self.env_args['sarsa_type'],
                credence_round=credence_round, lr=self.env_args['learning_rate'],
                n_track_adjust=lambda x: x / self.env_args['on_track'],
                learn_with_explore=self.env_args['learn_with_explore'],
                batch_size=self.env_args['sarsa_batch_size'], rolling_window=self.env_args['variance_window'],
                variance_type=self.env_args['variance_type'],
                do_variance=(self.env_args['voting'] == 'variance'),
                stochastic=self.env_args['stochastic_voting']
            )
            env_creator = lambda: model
        else:
            assert False

        return model, env_creator

    def _save_model_every(self, loc, glob):
        prev_timesteps = self.timesteps_so_far
        if loc is not None:
            self.timesteps_so_far = loc['self'].num_timesteps

        if prev_timesteps // self.env_args['checkpoint_timesteps'] != self.timesteps_so_far // self.env_args['checkpoint_timesteps'] and self.save_folder is not None:
            self.model.save(self.save_folder + f'/{self.timesteps_so_far:010}')

    def train_trolley(self, level='classic', on_track=10, on_track_dist='oneto', voting='nash',
                      theories=({"pushed_harms":-4,"collateral_harms":-1, 'lies': -0.5, 'doomsday': -10},{"harms": -1, 'doomsday': -300}),
                      credences=None, nenvs=32, seed=-1, num_timesteps=50000000, stochastic_voting=False,
                      cost_exponent=1, sarsa_type='deep', credence_granularity=20, learn_with_explore=False,
                      sarsa_eps=0.1, learning_rate=0.001, variance_window=None, sarsa_batch_size=32, save_to='results',
                      force_retry=False, variance_type='deep', n_sequential=1, checkpoint_timesteps=None, n_halves=10,
                      rand_adv=False):
        if checkpoint_timesteps is None:
            checkpoint_timesteps = num_timesteps // 20
        self.env_args = dict(
            level=level, on_track=on_track, on_track_dist=on_track_dist, voting=voting, theories=theories,
            stochastic_voting=stochastic_voting, cost_exponent=cost_exponent, sarsa_type=sarsa_type,
            credence_granularity=credence_granularity, credences=credences, seed=seed, nenvs=nenvs,
            learn_with_explore=learn_with_explore, sarsa_eps=sarsa_eps, learning_rate=learning_rate,
            variance_window=variance_window, sarsa_batch_size=sarsa_batch_size, variance_type=variance_type,
            n_sequential=n_sequential, checkpoint_timesteps=checkpoint_timesteps, n_halves=n_halves, rand_adv=rand_adv
        )

        model, env_creator = self._get_trolley_model(is_testing=False)

        self.save_folder = save_to
        cnt = 0
        while os.path.exists(self.save_folder + '/final_net.zip') and force_retry:
            # if os.path.exists(self.save_folder + '/final_net.zip') and not force_retry:
            #     print(
            #         f'It appears this job has already been completed in {self.save_folder} and --force_retry wasn\'t passed. Abandoning run.')
            #     return
            self.save_folder = f'{save_to}__retry-{cnt:02}'
            cnt += 1
        os.makedirs(self.save_folder)
        pickle.dump(self.env_args, open(self.save_folder + '/args.pickle', 'wb'))

        self.model = model
        model.learn(total_timesteps=num_timesteps, callback=self._save_model_every)

        if save_to is not None:
            model.save(self.save_folder + '/final_net')

    def test_trolley(self, load_from, n_credences=None, on_track_min=1, on_track_max=None,
                     n_on_track=None, sequence_number=0, filename='final_net', suffix_name=None):
        self.env_args = pickle.load(open(load_from + '/args.pickle', 'rb'))
        for filename in ['final_net'] + [e.split('/')[-1] for e in glob.glob(load_from + '/00*')]:
            model, env_creator = self._get_trolley_model(is_testing=True)
            model = model.load(load_from + '/' + filename)

            on_track_list = possible_values_dist(self.env_args['on_track_dist'], self.env_args['on_track'])
            if on_track_max or n_on_track:
                if on_track_max is None:
                    on_track_max = max(on_track_list)
                interval = (on_track_max - on_track_min) / n_on_track if n_on_track is not None else 1
                on_track_list = np.arange(on_track_min, on_track_max + interval, interval)
            self._test_trolley(
                model, env_creator, granularity=n_credences,
                on_track_list=on_track_list, sequence_number=sequence_number,
                filename=load_from + '/' + (
                    suffix_name + '__' + filename
                    if suffix_name is not None else
                    f'results__{filename}__credences-{n_credences}__on_track-{on_track_min}-{on_track_max}-{n_on_track}__seq-{sequence_number}'
                )
            )

    def _test_trolley(self, model, env_creator, granularity, on_track_list, sequence_number, filename):
        if os.path.exists(filename + '.png') and os.path.exists(filename + '.pdf'):
            return
        if granularity is None:
            granularity = self.env_args['credence_granularity']
        from collections import defaultdict
        total_map = None
        total_map = defaultdict(lambda: len(total_map))
        env = env_creator()
        outcome_map = {'value': [], 'Deontology Credence': [], '# On Track': []}
        outcome_pic = []
        colors = [[0x87, 0xAF, 0xFF], [0xFF, 0x8F, 0x49], [0xC0, 0xFF, 0x80], [0xA0, 0x00, 0xA0], [0, 0, 0]]
        possible_values = set()
        for cur_on_track in tqdm(on_track_list):
            outcome_pic.append([])
            for cred in range(granularity + 1):
                obs = env.reset(
                    np.array(
                        [cred / granularity,
                         (granularity - cred) / granularity]),
                    cur_on_track
                )
                cur_sequence = 0
                total = np.zeros(2 if self.env_args.get('rand_adv') else len(self.env_args['theories']))
                done = False
                total_uncaused = 0
                total_pushed = 0
                total_collateral = 0
                total_lies = 0
                total_doomsday = 0
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    if cur_sequence == sequence_number:
                        total_uncaused += info['uncaused_harms']
                        total_lies += info['lies']
                        total_pushed += info['pushed_harms']
                        total_collateral += info['collateral_harms']
                        total_doomsday += info['doomsday']
                    cur_sequence += info['subenv_done']
                    total += rewards
                outcome_map['value'].append(total_uncaused > 0)
                outcome_map['Deontology Credence'].append(cred / granularity)
                outcome_map['# On Track'].append(cur_on_track)
                if total_doomsday > 0:
                    code = 4
                elif total_lies > 0 and total_uncaused > 0:
                    code = 3
                elif total_uncaused > 0:
                    code = 0
                elif total_collateral > 0:
                    code = 1
                elif total_pushed > 0:
                    code = 2
                else:
                    assert False
                possible_values.add(code)
                outcome_pic[-1].append(colors[code])

        outcome_pic = np.array(outcome_pic)[::-1]
        labels = ['Nothing', 'Switch', 'Push', 'Lie Only', 'Doomsday']
        patches = [mpatches.Patch(color=np.array(colors[i]) / 255, label=labels[i]) for i in range(len(labels)) if i in possible_values]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches)
        plt.imshow(outcome_pic)
        def show_ticks(tick_fn, fake_min, fake_max, true_min, true_max, n_splits, true_formatter, reverse=False):
            fake_min -= 0.5
            fake_max += 0.5
            fake_interval = fake_max - fake_min
            true_interval = true_max - true_min
            fake_ticks = []
            true_ticks = []
            for i in range(0, n_splits + 1):
                fake_ticks.append(i * fake_interval / n_splits + fake_min)
                true_ticks.append(true_formatter(i * true_interval / n_splits + true_min))
            tick_fn(fake_ticks, list(reversed(true_ticks)) if reverse else true_ticks)
        show_ticks(plt.xticks, 0, granularity, 0, 100, 4, lambda x: f'{x:.0f}%')
        def good_div(v):
            v = int(np.round(v))
            divs = [i for i in range(1, v + 1) if v % i == 0]
            res = min(divs, key=lambda i: abs(v // i - 7))
            # print('OMG', res, [(i, abs(v // i - 7)) for i in divs])
            return v // res
        show_ticks(plt.yticks, 0, len(outcome_pic) - 1, on_track_list[0], on_track_list[-1], good_div(on_track_list[-1] - on_track_list[0]), lambda x: f'{x:.0f}', reverse=True)
        plt.xlabel('Credence in deontology')
        plt.ylabel('Number on tracks (X)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if filename is not None:
            plt.savefig(filename + '.png')
            plt.savefig(filename + '.pdf')
        # plt.show()


    def _mycallback_uniform(self, loc, glob):
        prev_calls = self.n_calls
        prev_timesteps = self.timesteps_so_far
        if loc is not None:
            self.timesteps_so_far = loc['self'].num_timesteps
            self.n_calls += 1

        should_checkpoint_timesteps =  (
                self.checkpoint_timesteps != -1 and (
                    self.timesteps_so_far == 0 or
                    prev_timesteps // self.checkpoint_timesteps != self.timesteps_so_far // self.checkpoint_timesteps
                )
        )
        should_checkpoint_calls =  (
                self.checkpoint_episodes != -1 and (
                    self.n_calls == 0 or
                    prev_calls // self.checkpoint_episodes != self.n_calls // self.checkpoint_episodes
                )
        )

        should_checkpoint = should_checkpoint_calls or should_checkpoint_timesteps

        filename = f'{self.save_folder}/episode-{self.n_calls:08}_step-{self.timesteps_so_far:012}'

        if self.save_folder and should_checkpoint:
            self.model.save(filename + '_net')
        if self.intermediate_test_episodes > 0 and should_checkpoint:
            self._test_uniform(self.intermediate_test_episodes, filename + '_test.pickle' if self.save_folder else None)

    def train_uniform(self, n_agents=2, n_actions=4, episode_steps=20, know_other_preferences=False, stochastic_voting=False,
              seed=-1, num_timesteps=10000, cost_exponent=2, mean_of_std=0.3, std_of_mean=0.1, nenvs=64,
              save_to=None, final_test_episodes=None, intermediate_test_episodes=0,
              checkpoint_timesteps=-1, checkpoint_episodes=-1, ent_coef=0.01, force_retry=False):
        self.env_args = dict(
            n_agents=n_agents, n_actions=n_actions, n_steps=episode_steps,
            know_other_preferences=know_other_preferences, stochastic_voting=stochastic_voting,
            cost_exponent=cost_exponent, mean_of_std=mean_of_std, std_of_mean=std_of_mean
        )

        self.intermediate_test_episodes = intermediate_test_episodes
        if final_test_episodes is None:
            final_test_episodes = intermediate_test_episodes
        self.save_folder = save_to
        cnt = 0
        while os.path.exists(self.save_folder):
            if os.path.exists(self.save_folder + '/final_net.zip') and not force_retry:
                print(f'It appears this job has already been completed in {self.save_folder} and --force_retry wasn\'t passed. Abandoning run.')
                return
            self.save_folder = f'{save_to}__retry-{cnt:02}'
            cnt += 1
        self.checkpoint_timesteps = checkpoint_timesteps
        self.checkpoint_episodes = checkpoint_episodes

        if self.save_folder:
            os.makedirs(self.save_folder)
            pickle.dump(self.env_args, open(self.save_folder + '/env_args.pickle', 'wb'))
            json.dump(sys.argv, open(self.save_folder + '/kwargs.json', 'w'), indent=2)


        env = DummyVecEnv([lambda: PreferenceEnv(**self.env_args)] * nenvs)
        self.model = PPO2("MlpPolicy", env, verbose=1, seed=seed if seed > 0 else None, ent_coef=ent_coef)
        self._mycallback_uniform(None, None)
        self.model.learn(total_timesteps=num_timesteps, callback=self._mycallback_uniform)

        if self.save_folder:
            self.model.save(self.save_folder + '/final_net')

        if final_test_episodes is not None and final_test_episodes > 0:
            self._test_uniform(
                test_episodes=final_test_episodes, save_to=self.save_folder + '/final_test.pickle'
            )

    def _test_uniform(self, test_episodes, save_to):
        env = PreferenceEnv(**self.env_args)
        obs = env.reset()
        total = np.zeros(self.env_args['n_agents'])
        max_possible = np.zeros(self.env_args['n_agents'])
        data = []
        episode_data = {'obs': [], 'actions': [], 'rewards': [], 'done': []}
        while len(data) < test_episodes:
            action, _states = self.model.predict(obs, deterministic=True)
            max_possible += np.max(obs[:, :self.env_args['n_actions']], axis=1)
            episode_data['obs'].append(obs)
            episode_data['actions'].append(action)
            obs, rewards, done, info = env.step(action)
            episode_data['rewards'].append(rewards)
            episode_data['done'].append(done)
            total += rewards
            if done:
                for k in episode_data:
                    episode_data[k] = np.array(episode_data[k])
                data.append(episode_data)
                episode_data = {'obs': [], 'actions': [], 'rewards': [], 'done': []}
                obs = env.reset()

        print(max_possible, total)

        if save_to:
            gzip.open(save_to + '.gz', 'wb').write(pickletools.optimize(pickle.dumps(data)))

    def test_uniform(self, load_from, save_to=None, test_episodes=1000, **kwargs):
        self.env_args = pickle.load(open('/'.join(load_from.split('/')[:-1]) + '/env_args.pickle', 'rb'))
        for k in kwargs:
            if k.startswith('override_'):
                self.env_args[k[len('override_'):]] = kwargs[k]
        self.model = PPO2.load(load_from)
        self._test_uniform(test_episodes, save_to)

if __name__ == '__main__':
    fire.Fire(FreeformVoter)

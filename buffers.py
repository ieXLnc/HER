import numpy as np
import random
import torch
import gym
import copy
from utils import *

torch.manual_seed(14)
# set GPU for faster training
cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class ReplayBuffer:
    def __init__(self, size):
        self.max_size = size
        self._storage = []
        self._next_idx = 0

    def add(self, obs, action, reward, obs_tp1, done):
        data = (obs, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self.max_size

    def _return_sample(self, idxes):
        obses, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for idx in idxes:
            data = self._storage[idx]
            obs, action, reward, obs_tp1, done = data
            obses.append(np.array(obs))
            actions.append(np.array(action))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1))
            dones.append(done)

        return obses, actions, rewards, obses_tp1, dones

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) -1) for _ in range(batch_size)]
        return self._return_sample(idxes)


# HER REPLAY BUFFER
class HindsightExperienceReplay:
    def __init__(self, replay_buffer, n_sampled_goals, wrapped_env):

        self.n_sampled_goals = n_sampled_goals
        self.env = wrapped_env
        self.replay_buffer = replay_buffer
        self.episode_transitions = []

    def add(self, obs, action, reward, obs_tp1, done, info):
        """
        add a new transition to the buffer

        :param obs: (np array) current obs
        :param action: ([float]) action taken
        :param reward: (float) reward got for the action
        :param obs_tp1: (np.array) next obs after the action
        :param done: (bool) if the ep is over
        :param info: (dict) extra info to calculate rewards later
        """
        # add data to the current ep
        self.episode_transitions.append((obs, action, reward, obs_tp1, done, info))
        if done:
            # add real and imagined transitions to the replay buffer
            self._store_episode()
            # reset ep buffer
            self.episode_transitions = []

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        sample an achieved goal based on the strategy: currently only future

        :param episode_transitions: (list) current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.array) an achieved goal
        """
        # sample a goal that was observed during the same ep after the current step --> FUTURE STRAT
        selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
        selected_transition = episode_transitions[selected_idx]
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        sample a batch based on the sampling strategy

        :param episode_transitions: (list) current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.array) multiples achieved goals
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx) for _ in range(self.n_sampled_goals)
        ]

    def _store_episode(self):
        """
        Sample new goal and add them in the replay buffer
        Only called at the end of the ep
        """
        # for each transition in the last ep
        # create a new set of artificial ones
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs, action, reward, obs_tp1, done, info = transition

            # add the last ep to the replay buffer
            self.replay_buffer.add(obs, action, reward, obs_tp1, done)

            # if future we cannot sample a goal that is at the last step of the ep
            if transition_idx == len(self.episode_transitions) - 1:
                break

            # sample n goals per transitions, n='n_sampled_goals'
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # for each sampled goals store a new transition
            for goal in sampled_goals:

                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
                # convert concat obs to dict to update goals
                obs_d, next_obs_d = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # update desire goal in the transition
                obs_d['desired_goal'] = goal
                next_obs_d['desired_goal'] = goal

                # update the reward based on the new goal
                reward = self.env.compute_reward(next_obs_d['achieved_goal'], goal, info)
                done = False

                # transform back to array
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_d, next_obs_d))

                # add the artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)


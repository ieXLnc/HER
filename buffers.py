import threading
import numpy as np


# https://github.com/TianhongDai/hindsight-experience-replay/blob/master/rl_modules/replay_buffer.py
class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']            # max timesteps in one ep ?
        self.size = buffer_size // self.T
        self.sample_func = sample_func
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']])
                        }

        # thread lock
        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_action = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store info
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_action
        self.n_transitions_stored += self.T * batch_size

    def sample(self, batch_size):
        temp_buffer = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffer[key] = self.buffers[key][:self.current_size]    # get all current obs
        temp_buffer['obs_next'] = temp_buffer['obs'][:, 1:, :]
        temp_buffer['ag_next'] = temp_buffer['ag'][:, 1:, :]
        # sample transition
        transitions = self.sample_func(temp_buffer, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx


# HER REPLAY BUFFER
class HERSample:
    def __init__(self, replay_strategy='future', n_sample=4, reward_func=None):
        self.replay_strategy = replay_strategy
        self.n_sample = n_sample
        if self.replay_strategy == 'future':
            self.her_ratio = 1 - (1. / (1 + self.n_sample))
        else:
            self.her_ratio = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_transitions):
        T = episode_batch['actions'].shape[1]                       # T = 50, lenght of ep
        rollout_batch_size = episode_batch['actions'].shape[0]
        # print('roolout batch size', rollout_batch_size)           # get all the rollouts
        batch_size = batch_transitions                              # batch size sans surprise
        # select which rollout and timesteps to be used
        episodes_idxs = np.random.randint(0, rollout_batch_size, batch_size)    # getting random ep numbers
        t_samples = np.random.randint(T, size=batch_size)                       # take t samples in T size (batch_size, )
        # select those random t_sample sequences in the randomly selected ep
        transitions = {key: episode_batch[key][episodes_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her indxs
        her_idx = np.where(np.random.uniform(size=batch_size) < self.her_ratio)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_idx]
        # replace goals with achieved goals
        future_ag = episode_batch['ag'][episodes_idxs[her_idx], future_t]
        transitions['g'][her_idx] = future_ag
        # get the params to recompute rewards with env.compute rewards func
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from networks import *
from utils import *
from buffers import *
import pickle
import matplotlib.pyplot as plt

torch.manual_seed(14)
# set GPU for faster training
cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class DDPGAgent:
    def __init__(
            self,
            env,
            env_params,
            fc1=256, fc2=256, fc3=256,
            gamma=0.98, tau=0.95,
            actor_lr=0.001, critic_lr=0.001,
            batch_size=128,
            clip_range=5
    ):

        self.env = env
        self.env_params = env_params

        # hyperparams
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.action_l2 = 1
        self.tau = tau
        self.clip_range = clip_range
        self.clip_ratio = 200

        # Create the networks
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

        self.actor = Actor(self.env_params['obs'] + self.env_params['goal'], self.fc1, self.fc2, self.fc3, self.env_params['action']).to(device)
        self.actor_target = Actor(self.env_params['obs'] + self.env_params['goal'], self.fc1, self.fc2, self.fc3, self.env_params['action']).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.env_params['obs'] + self.env_params['goal'], self.env_params['action'], self.fc1, self.fc2, self.fc3).to(device)
        self.critic_target = Critic(self.env_params['obs'] + self.env_params['goal'], self.env_params['action'], self.fc1, self.fc2, self.fc3).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

        # normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.clip_range)

        # setup weights
        for target_params, params in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_params.data.copy_(params.data)
        for target_params, params in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_params.data.copy_(params.data)

        # Memory
        self.memory_size = 100_000
        self.batch_size = batch_size
        self.her_sample = HERSample(replay_strategy='future',
                                    n_sample=5,
                                    reward_func=self.env.compute_reward)
        self.replay_buffer = ReplayBuffer(env_params=self.env_params,
                                         buffer_size=self.memory_size,
                                         sample_func=self.her_sample.sample_her_transition)

        self.n_epochs = 100
        self.n_cycles = 50
        self.rollout = 16
        self.n_update = 40
        self.test_rollout = 5

        # Noise
        self.noise_eps = 0.2        # add noise with 0.2
        self.random_eps = 0.1       # choose action over eps

        # saving modalities
        self.name_env = env.unwrapped.spec.id
        self.name = self.name_env + '_model' + '.pth'

        # Create log
        self.log = {
            'success': [],
            'actor_loss': [0],
            'critic_loss': [0],
            'epoch': [],
            'batch_size': self.batch_size,
        }

    def preprocess(self, observation, goal):
        # normalize when created the norm pipe
        obs_norm = self.o_norm.normalize(observation)
        g_norm = self.g_norm.normalize(goal)
        # concat
        input_obs = np.concatenate([obs_norm, g_norm])
        input_obs_t = torch.tensor(input_obs, dtype=torch.float)
        return input_obs_t

    def clip_input(self, obs, g):
        obs = np.clip(obs, -self.clip_ratio, self.clip_ratio)
        g = np.clip(g, -self.clip_ratio, self.clip_ratio)
        return obs, g

    def get_action(self, input):
        # print(input, input.shape)
        if isinstance(input, np.ndarray):
            input = torch.FloatTensor(input).to(device)

        action = self.actor(input).detach().cpu().numpy()
        #
        # # add the gaussian
        # action += self.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        # action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # # random actions...
        # random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
        #                                     size=self.env_params['action'])
        # # choose if use the random actions
        # action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
        return action

    def update_models(self):

        # sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        # print('transition:', transitions)
        # print('keys:', transitions.keys())      # 'obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next', 'r'
        # get values out of transitions
        obs, obs_next, g = transitions['obs'], transitions['obs_next'], transitions['g']    # get the vals out to clip
        transitions['obs'], transitions['g'] = self.clip_input(obs, g)
        transitions['obs_next'], transitions['g_next'] = self.clip_input(obs_next, g)
        # normalize the vals and concat
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        input = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        input_next = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # tensor those + r and actions
        input_t = torch.tensor(input, dtype=torch.float)
        input_next_t = torch.tensor(input_next, dtype=torch.float)
        rew_t = torch.tensor(transitions['r'], dtype=torch.float)
        actions_t = torch.tensor(transitions['actions'], dtype=torch.float)

        # -------- Calculate losses --------
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target.forward(input_next_t)
            next_q_value = self.critic_target.forward(input_next_t, next_actions.detach())
            next_q_value = next_q_value.detach()
            target_q = rew_t * self.gamma * next_q_value
            target_q = target_q.detach()
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            target_q = torch.clamp(target_q, -clip_return, 0)   # clip target used to train critic as in the paper

        q_vals = self.critic.forward(input_t, actions_t)
        critic_loss = nn.MSELoss()(target_q, q_vals)
        self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())

        # Actor loss
        action_r = self.actor.forward(input_t)
        policy_loss = - self.critic.forward(input_t, action_r).mean()
        policy_loss += self.action_l2 * (action_r / self.env_params['action_max']).pow(2).mean()     # L2 reg
        self.log['actor_loss'].append(policy_loss.detach().cpu().numpy())

        # -------- Update networks --------
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def soft_update(self, target_net, net):     # creation of a separate soft def to update every n step
        # -------- Soft update of nets --------
        for target_params, params in zip(target_net.parameters(), net.parameters()):
            target_params.data.copy_(self.tau * params.data + (1.0 - self.tau) * target_params.data)

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_sample.sample_her_transition(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self.clip_input(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def save_models(self):
        torch.save(self.actor.state_dict(), './Models/' + 'actor_' + self.name)
        torch.save(self.actor_target.state_dict(), './Models/' + 'actor_target_' + self.name)
        torch.save(self.critic.state_dict(), './Models/' + 'critic_' + self.name)
        torch.save(self.critic_target.state_dict(), './Models/' + 'critic_target_' + self.name)

    def train(self):

        for epoch in range(self.n_epochs):

            for _ in range(self.n_cycles):

                mb_obs, mb_ag, mb_g, mb_act = [], [], [], []

                for _ in range(self.rollout):
                    # reset rollout
                    ep_obs, ep_ag, ep_g, ep_act = [], [], [], []
                    # reset env
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start collecting sample
                    for t in range(self.env_params['max_timesteps']):

                        input_obs = self.preprocess(obs, g)             # normalize and concat
                        action = self.get_action(input_obs)             # remove step because gaussian

                        # feed action
                        new_observation, _, _, info = self.env.step(action)
                        obs_new = new_observation['observation']
                        ag_new = new_observation['achieved_goal']
                        # append rollout
                        ep_obs.append(obs)
                        ep_ag.append(ag)
                        ep_g.append(g)
                        ep_act.append(action)
                        # new vals for obs and ag
                        obs = obs_new
                        ag = ag_new

                    # append the last vals
                    ep_obs.append(obs)
                    ep_ag.append(ag)
                    # record cycle
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_act.append(ep_act)

                # convert to array
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_act = np.array(mb_act)

                # store cycle
                self.replay_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_act])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_act])
                # self.update norm

                for _ in range(self.n_update):
                    self.update_models()

                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic_target, self.critic)

            mean_success = self.eval_agent()
            self.log['success'].append(mean_success)
            self.log['epoch'].append(epoch)
            self.summary()
            if mean_success > 0.8:
                self.save_models()
                with open('./Models/logger_' + self.name + '.pkl', 'wb') as handle:
                    pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if epoch >= self.n_epochs - 1:
                self.save_models()
                with open('./Models/logger_' + self.name + '.pkl', 'wb') as handle:
                    pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
                plt.plot(self.log['success'])
                plt.savefig('./Plots/plot_success_fetch.png')

    def eval_agent(self, render=False, n_test=1):
        total_success = []
        for _ in range(self.test_rollout):
            success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if render:
                    self.env.render()
                input_t = self.preprocess(obs, g)
                action = self.actor(input_t)
                new_observation, _, _, info = self.env.step(action.detach().numpy())
                # replace with new vals
                obs = new_observation['observation']
                g = new_observation['desired_goal']
                # append success rate for the ep
                success_rate.append(info['is_success'])

            total_success.append(success_rate)

        mean_success = np.mean(total_success)

        return mean_success

    def summary(self):
        print(f'-------------------------------------------------')
        print(f'----------- Episode #{self.log["epoch"][-1]}-------------------')
        print(f'Mean success: {self.log["success"][-1]}')
        print(f'Actor loss: {self.log["actor_loss"][-1]}')
        print(f'Critic loss: {self.log["critic_loss"][-1]}')
        print(f'With Batch size of {self.log["batch_size"]}')
        if self.log["epoch"][-1] >= self.n_epochs-1:
            self.save_models()
            with open('./Models/logger_' + self.name + '.pkl', 'wb') as handle:
                pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'-------------------------------------------------')

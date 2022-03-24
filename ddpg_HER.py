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
        # epochs and other
        self.n_epochs = 150
        self.n_cycles = 50
        self.rollout = 16
        self.n_update = 40
        self.test_rollout = 10
        # Noise
        self.noise_eps = 0.2        # add noise with 0.2
        self.random_eps = 0.3       # choose action over eps

        # Create the networks
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

        self.input_mod = self.env_params['obs'] + self.env_params['goal'] + self.env_params['goal']
        self.actor = Actor(self.input_mod, self.fc1, self.fc2, self.fc3, self.env_params['action']).to(device)
        self.actor_target = Actor(self.input_mod, self.fc1, self.fc2, self.fc3, self.env_params['action']).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.input_mod, self.env_params['action'], self.fc1, self.fc2, self.fc3).to(device)
        self.critic_target = Critic(self.input_mod, self.env_params['action'], self.fc1, self.fc2, self.fc3).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

        # setup weights
        for target_params, params in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_params.data.copy_(params.data)
        for target_params, params in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_params.data.copy_(params.data)

        # Memory
        self.memory_size = 1_000_000
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(size=self.memory_size)
        self.HER = HindsightExperienceReplay(self.replay_buffer, 4, self.env)

        # saving modalities
        self.name_env = 'test_for_now_1'
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
        # concat
        input_obs = np.concatenate([observation, goal])
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
        obses, actions, rewards, obses_tp1, dones = transitions         # len(256)
        # transform to tensors
        obses_t = torch.tensor(obses, dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.float)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        next_obses_t = torch.tensor(obses_tp1, dtype=torch.float)
        # -------- Calculate losses --------
        # Critic loss
        # with torch.no_grad():
        next_actions = self.actor_target.forward(next_obses_t)
        next_q_value = self.critic_target.forward(next_obses_t, next_actions.detach())
        target_q = rewards_t * self.gamma * next_q_value  # removed the detach from targetq and nextq
        # clip the q value
        clip_return = 1 / (1 - self.gamma)
        target_q = torch.clamp(target_q, -clip_return, 0)   # clip target used to train critic as in the paper

        q_vals = self.critic.forward(obses_t, actions_t)
        critic_loss = (target_q - q_vals).pow(2).mean()  # nn.MSELoss()(target_q, q_vals)
        self.log['critic_loss'].append(critic_loss.detach().cpu().numpy())

        # Actor loss
        action_r = self.actor.forward(obses_t)
        policy_loss = - self.critic.forward(obses_t, action_r).mean()
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

    def save_models(self):
        torch.save(self.actor.state_dict(), './Models/' + 'actor_' + self.name)
        torch.save(self.actor_target.state_dict(), './Models/' + 'actor_target_' + self.name)
        torch.save(self.critic.state_dict(), './Models/' + 'critic_' + self.name)
        torch.save(self.critic_target.state_dict(), './Models/' + 'critic_target_' + self.name)

    def train(self):

        for epoch in range(self.n_epochs):

            for _ in range(self.n_cycles):

                for _ in range(self.rollout):
                    # reset env
                    obs = self.env.reset()
                    # print('observation ? :', observation)
                    # obs = observation['observation']
                    # ag = observation['achieved_goal']
                    # g = observation['desired_goal']
                    # start collecting sample
                    for t in range(self.env_params['max_timesteps']):

                        action = self.get_action(obs)
                        # feed action
                        new_obs, reward, done, info = self.env.step(action)
                        # new vals for obs and ag
                        obs = new_obs

                        self.HER.add(obs, action, reward, new_obs, done, info)
                        # learn
                        if len(self.replay_buffer._storage) >= self.batch_size:
                            self.update_models()

                # learnings loop
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
            obs = self.env.reset()
            # obs = observation['observation']
            # g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if render:
                    self.env.render()
                # input_t = self.preprocess(obs, g)
                obs_t = torch.tensor(obs, dtype=torch.float)
                action = self.actor(obs_t)
                new_obs, _, _, info = self.env.step(action.detach().numpy())
                # replace with new vals
                obs = new_obs
                success_rate.append(info['is_success'])

            total_success.append(success_rate)

        total_success = np.array(total_success)
        local_success_rate = np.mean(total_success[:, -1])

        return local_success_rate

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

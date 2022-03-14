import numpy as np
import torch # Torch version :1.9.0+cpu
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
            fc1=64, fc2=64, fc3=64,
            gamma=0.95, tau=0.01,
            actor_lr=0.01, critic_lr=0.001,
            batch_size=64,
            noise=None,
            noise_std=0.3,
    ):

        self.env = env
        self.n_obs = get_obs_shape(self.env.observation_space)["observation"][0]
        self.n_obs_g = get_obs_shape(self.env.observation_space)["desired_goal"][0]
        self.n_obs_ag = get_obs_shape(self.env.observation_space)["achieved_goal"][0]
        self.n_acts = env.action_space.shape[0]
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.env_params = env_params

        # hyperparams
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.action_l2 = 1
        self.tau = tau
        self.min = -np.inf
        self.max = np.inf
        self.learning_steps = 1000

        # Create the networks
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

        self.actor = Actor(self.n_obs + self.n_obs_g, self.fc1, self.fc2, self.fc3, self.n_acts).to(device)
        self.actor_target = Actor(self.n_obs + self.n_obs_g, self.fc1, self.fc2, self.fc3, self.n_acts).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.n_obs + self.n_obs_g, self.n_acts, self.fc1, self.fc2, self.fc3).to(device)
        self.critic_target = Critic(self.n_obs + self.n_obs_g, self.n_acts, self.fc1, self.fc2, self.fc3).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

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
        self.noise_type = noise
        self.noise_std = noise_std
        if self.noise_type is not None:
            if self.noise_type == 'ou':
                self.OU_noise = OUNoise(self.env.action_space, max_sigma=self.noise_std)
                print('Ou noise used')

        # saving modalities
        self.name_env = env.unwrapped.spec.id
        self.name = self.name_env + '_model_' + str(self.noise_type) + '_' + str(self.noise_std) + '_normalize_' \
                    + '.pth'

        # Create log
        self.log = {
            'rewards': [],
            'rewards_ep': -np.inf,
            'mean_rewards': [],
            'best_score': [],
            'actor_loss': [0],
            'critic_loss': [0],
            'episode': 0,
            'batch_size': self.batch_size,
            'test_rew': [],
            'timesteps': 0
        }

    def get_action(self, input):
        # print(input, input.shape)
        if isinstance(input, np.ndarray):
            input = torch.FloatTensor(input).to(device)

        action = self.actor(input).detach().cpu().numpy()

        # implement gaussian noise and clip here

        return np.clip(action, self.low, self.high)

    def update_models(self):

        # sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        # print('transition:', transitions)
        # print('keys:', transitions.keys())      # 'obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next', 'r'
        # get values out of transitions
        obs = transitions['obs']
        obs_next = transitions['obs_next']
        g = transitions['g']

        # clips the vals ??
        # normalize the vals

        # concat obs + goals for input
        input = np.concatenate([obs, g], axis=1)
        input_next = np.concatenate([obs_next, g], axis=1)
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

    def save_models(self):
        torch.save(self.actor.state_dict(), './Models/' + 'actor_' + self.name)
        torch.save(self.actor_target.state_dict(), './Models/' + 'actor_target_' + self.name)
        torch.save(self.critic.state_dict(), './Models/' + 'critic_' + self.name)
        torch.save(self.critic_target.state_dict(), './Models/' + 'critic_target_' + self.name)
        # self.replay_buffer.save_memory(self.name)

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
                # self.update norm

                for _ in range(self.n_update):
                    self.update_models()

                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic_target, self.critic)

            mean_success = self.eval_agent()
            print(f'Mean success of epoch {epoch}: {mean_success}')

    def preprocess(self, observation, goal):
        # normalize when created the norm pipe
        # concat
        input_obs = np.concatenate([observation, goal])
        return input_obs

    def eval_agent(self, render=False, n_test=1):
        total_success = []
        for _ in range(self.test_rollout):
            success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                input = np.concatenate([obs, g])
                input_t = torch.tensor(input, dtype=torch.float)
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
        # if len(self.log['rewards']) == 0:
        #     best_current_score = -np.inf
        # else:
        #     best_current_score = self.log['rewards'][np.argmax(self.log['rewards'])]

        # last_val = self.log['rewards_ep']

        # self.log['rewards'].append(self.log['rewards_ep'])
        # self.log['mean_rewards'].append(np.mean(self.log['rewards'][-10:]))
        # mean_early_stop = np.mean(self.log['test_rew'][-100:])

        print(f'-------------------------------------------------')
        print(f'----------- Episode #{self.log["episode"]}-------------------')
        print(f'Rewards for the episode: {self.log["rewards_ep"]}')
        # print(f'Mean value for last 10 {self.log["mean_rewards"][-1]}')
        # if last_val > best_current_score:
        #     self.save_models()
        #     with open('./Models/logger_' + self.name + '.pkl', 'wb') as handle:
        #         pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     print(f'New models saved with {last_val}')
        # else:
        #     print(f'Best model: {best_current_score}')

        # print(f'test rewards: {self.log["test_rew"][-1]}')
       #  print(f'mean early stop is currently: {mean_early_stop}')
        print(f'Actor loss: {self.log["actor_loss"][-1]}')
        print(f'Critic loss: {self.log["critic_loss"][-1]}')
        print(f'With Batch size of {self.log["batch_size"]}')
        print(f'Timesteps: {self.log["timesteps"]}')

        # if mean_early_stop > self.early_stop_val or self.log["timesteps"] >= self.early_stop_timesteps:
        #     self.early_stop = True
        #     self.save_models()
        #     with open('./Models/logger_' + self.name + '.pkl', 'wb') as handle:
        #         pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     print(f'Early stop activated with score {last_val} at episode {self.log["episode"]}')
        print(f'-------------------------------------------------')

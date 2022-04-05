import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Create both Actor and Critic network
Both take obs['obervation'] and observation['desired_goals'] as input, the critic also takes the actions
"""


# Creates the Actor class
class Actor(nn.Module):
    def __init__(self, env_params, fc_shape):
        super(Actor, self).__init__()
        self.fc = fc_shape
        self.obs_shape = env_params['obs']
        self.goal_shape = env_params['goal']
        self.action_dims = env_params['action']
        self.action_max = env_params['action_max']

        self.l1 = nn.Linear(self.obs_shape + self.goal_shape, self.fc)
        self.l2 = nn.Linear(self.fc, self.fc)
        self.l3 = nn.Linear(self.fc, self.fc)
        self.l4 = nn.Linear(self.fc, self.action_dims)

    def forward(self, obs_g):
        x = F.relu(self.l1(obs_g))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        acts = self.action_max * torch.tanh(self.l4(x))

        return acts


# Create the Critic class
class Critic(nn.Module):
    def __init__(self, env_params, fc_shape):
        super(Critic, self).__init__()
        self.fc = fc_shape
        self.obs_shape = env_params['obs']
        self.goal_shape = env_params['goal']
        self.action_dims = env_params['action']
        self.action_max = env_params['action_max']

        self.l1 = nn.Linear(self.obs_shape + self.goal_shape + self.action_dims, self.fc)
        self.l2 = nn.Linear(self.fc, self.fc)
        self.l3 = nn.Linear(self.fc, self.fc)
        self.l4 = nn.Linear(self.fc, 1)

    def forward(self, obs_g, actions):
        x = torch.cat([obs_g, actions / self.action_max], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        q_val = self.l4(x)

        return q_val









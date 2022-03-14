import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(14)
np.random.default_rng(14)

cuda = torch.cuda.is_available()  # check for CUDA
device = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))


class Critic(nn.Module):
    def __init__(self, in_dims, n_acts, fc1, fc2, fc3):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(in_dims, fc1)
        self.layer2 = nn.Linear(fc1 + n_acts, fc2)
        self.layer3 = nn.Linear(fc2, fc3)
        self.layer4 = nn.Linear(fc3, 1)

    def forward(self, state, action):

        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(torch.cat([x, action], 1))
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_dims, fc1, fc2, fc3, out_dims):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(in_dims, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, fc3)
        self.layer4 = nn.Linear(fc3, out_dims)

    def forward(self, state):

        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = torch.tanh(self.layer4(x))
        x = x

        return x

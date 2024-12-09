import torch
import torch.nn as nn
from torch.distributions import Normal

STATE_DIM = 189
HIDDEN_DIM = 256
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class SACActor(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(SACActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()

        return std, mean

    def sample(self, state):
        std, mean = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        # Compute log probability, using the formula for transformed distribution
        log_prob = normal.log_prob(x_t)

        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, hidden_dim):
        super(SACCritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class QNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

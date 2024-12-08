from torch import nn
import torch.nn.functional as f
from torch.distributions.categorical import Categorical


ACTION_DIM = 5
STATE_DIM = 189
HIDDEN_DIM = 256


class Actor(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM)
        )

    def forward(self, state):
        return f.softmax(self.net(state), dim=-1)

    def get_action_probs(self, state):
        return self.forward(state)

    def sample_action(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


class Critic(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

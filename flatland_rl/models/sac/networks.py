from torch import nn, clamp, tanh, cat
from torch.distributions.normal import Normal


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SACActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = clamp(log_std, -20, 2)  # Prevent numerical instability
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = tanh(x_t)
        return action, dist.log_prob(x_t)


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SACCritic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = cat([state, action], dim=-1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        return q1, q2

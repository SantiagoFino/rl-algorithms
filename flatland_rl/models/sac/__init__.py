import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from rl_algorithms.flatland_rl.models.sac.networks import SACActor, SACCritic, QNetwork
from rl_algorithms.flatland_rl.utils import preprocess_state, create_flatland_env


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        batch_data = [self.buffer[i] for i in batch]

        # Convert to numpy arrays first
        states = np.vstack([data[0] for data in batch_data])
        actions = np.vstack([data[1] for data in batch_data])
        rewards = np.array([data[2] for data in batch_data])
        next_states = np.vstack([data[3] for data in batch_data])
        dones = np.array([data[4] for data in batch_data])

        # Then convert to tensors
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class SACMaster:
    def __init__(self, env_config, hidden_dim=256, action_dim=5):
        self.env_config = env_config
        self.env = create_flatland_env(**env_config)
        self.action_dim = action_dim

        # Initialize networks
        self.actor = SACActor(hidden_dim, action_dim)
        self.critic = SACCritic(hidden_dim)
        self.q_net1 = QNetwork(hidden_dim, action_dim)
        self.q_net2 = QNetwork(hidden_dim, action_dim)
        self.q_net1_target = QNetwork(hidden_dim, action_dim)
        self.q_net2_target = QNetwork(hidden_dim, action_dim)

        self.q_net1_target.load_state_dict(self.q_net1.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)
        self.q_net1_optimizer = Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_net2_optimizer = Adam(self.q_net2.parameters(), lr=3e-4)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training metrics
        self.rewards_history = []
        self.steps_history = []

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            _, mean = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)

        # Convert continuous action to discrete
        # Map from [-1,1] to [0,4]
        action = action.detach().cpu().numpy()[0]
        continuous_value = action[0]
        # Map from [-1,1] to [0,4]
        discrete_action = int((continuous_value + 1) * 2.5 % 5)

        return discrete_action if evaluate else discrete_action, action

    def update_parameters(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Update Q networks
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            next_q1 = self.q_net1_target(next_states, next_actions)
            next_q2 = self.q_net2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * next_q

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.q_net1_optimizer.zero_grad()
        q1_loss.backward()
        self.q_net1_optimizer.step()

        self.q_net2_optimizer.zero_grad()
        q2_loss.backward()
        self.q_net2_optimizer.step()

        # Update actor
        actions_pred, log_pi = self.actor.sample(states)
        q1_pi = self.q_net1(states, actions_pred)
        q2_pi = self.q_net2(states, actions_pred)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.q_net1.parameters(), self.q_net1_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for param, target_param in zip(self.q_net2.parameters(), self.q_net2_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def train(self, num_episodes, batch_size=256, updates_per_step=5):
        """Train the SAC agent"""
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = preprocess_state(obs[0])
            episode_reward = 0
            steps = 0
            done = False

            while not done:
                if info['action_required'][0]:
                    discrete_action, continuous_action = self.select_action(state)
                    action_dict = {0: discrete_action}
                else:
                    action_dict = {}

                next_obs, reward, dones, info = self.env.step(action_dict)
                next_state = preprocess_state(next_obs[0])
                done = dones['__all__']

                if 0 in action_dict:
                    # Store the continuous action in the buffer
                    self.replay_buffer.push(state, continuous_action, reward[0], next_state, float(done))
                    episode_reward += reward[0]

                if len(self.replay_buffer) > batch_size:
                    for _ in range(updates_per_step):
                        self.update_parameters(batch_size)

                state = next_state
                steps += 1

            self.rewards_history.append(episode_reward)
            self.steps_history.append(steps)

            print(f"\rEpisode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {steps}", end="")

        print("\nTraining completed!")

    def evaluate(self, num_episodes):
        """Evaluate the trained agent"""
        eval_rewards = []
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            state = preprocess_state(obs[0])
            episode_reward = 0
            done = False

            while not done:
                if info['action_required'][0]:
                    action = self.select_action(state, evaluate=True)
                    action_dict = {0: int(action)}
                else:
                    action_dict = {}

                next_obs, reward, dones, info = self.env.step(action_dict)
                next_state = preprocess_state(next_obs[0])
                done = dones['__all__']

                if 0 in action_dict:
                    episode_reward += reward[0]

                state = next_state

            eval_rewards.append(episode_reward)

        return np.mean(eval_rewards)

    def save_model(self, path):
        """Save the trained models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'q_net1_state_dict': self.q_net1.state_dict(),
            'q_net2_state_dict': self.q_net2.state_dict(),
        }, path)

    def load_model(self, path):
        """Load trained models"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.q_net1.load_state_dict(checkpoint['q_net1_state_dict'])
        self.q_net2.load_state_dict(checkpoint['q_net2_state_dict'])

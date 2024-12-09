import time
import torch
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import numpy as np

from rl_algorithms.flatland_rl.models.a3c.networks import Actor, Critic
from rl_algorithms.flatland_rl.utils import preprocess_state, create_flatland_env

from rl_algorithms.flatland.core.grid.grid4 import Grid4Transitions


class A3CWorker(mp.Process):
    def __init__(self, worker_id, global_actor, global_critic, env_config,
                 shared_optimizer_actor, shared_optimizer_critic, shared_episode_counter,
                 episode_lock, max_episodes, shared_metrics_queue):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id

        # Create local environment
        self.env = create_flatland_env(**env_config)

        # Create local networks
        self.local_actor = Actor()
        self.local_critic = Critic()
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.shared_optimizer_actor = shared_optimizer_actor
        self.shared_optimizer_critic = shared_optimizer_critic

        self.shared_episode_counter = shared_episode_counter
        self.episode_lock = episode_lock
        self.shared_metrics_queue = shared_metrics_queue

        # Hyperparameters
        self.max_episodes = max_episodes
        self.gamma = 0.99
        self.t_max = 5  # Local update interval

    def sync_with_global(self):
        """Sync local networks with global networks"""
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def compute_returns(self, rewards, values, dones):
        """Compute returns and advantages for policy update"""
        returns = []
        advantages = []
        R = 0 if dones[-1] else values[-1]

        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            advantage = R - v
            returns.append(R)
            advantages.append(advantage)

        return list(reversed(returns)), list(reversed(advantages))

    def run_episode(self):
        """Run a single episode"""
        self.sync_with_global()
        obs, info = self.env.reset()

        state = preprocess_state(obs[0])  # Process for single agent
        done = False

        # Storage for episode data
        states, actions, rewards, values, dones = [], [], [], [], []
        episode_reward = 0
        steps = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get valid actions
            if self.env.agents[0].position is not None:
                current_cell = self.env.rail.grid[self.env.agents[0].position[0]][self.env.agents[0].position[1]]
                valid_actions = Grid4Transitions.get_entry_directions(current_cell)
                valid_actions = [idx for idx, valid in enumerate(valid_actions) if valid]
            else:
                valid_actions = [2]

            # Get action from local network
            if info['action_required'][0]:
                action_probs = self.local_actor(state_tensor)
                # Mask invalid actions
                mask = torch.zeros_like(action_probs)
                mask[0, valid_actions] = 1
                masked_probs = action_probs * mask
                masked_probs = masked_probs / masked_probs.sum()

                dist = Categorical(masked_probs)
                action = dist.sample().item()
                action_dict = {0: action}
            else:
                action_dict = {}

            # Get value estimate
            value = self.local_critic(state_tensor)

            # Take action in environment
            next_obs, reward, dones_dict, info = self.env.step(action_dict)
            next_state = preprocess_state(next_obs[0])
            done = dones_dict['__all__']

            # Store transition
            if 0 in action_dict:  # Only store if we took an action
                states.append(state)
                actions.append(action)
                rewards.append(reward[0])
                values.append(value.item())
                dones.append(done)

            state = next_state
            episode_reward += reward[0] if 0 in reward else 0
            steps += 1

            # Perform update every t_max steps or at end of episode
            if len(states) == self.t_max or done:
                self.update_global_networks(states, actions, rewards, values, dones)
                states, actions, rewards, values, dones = [], [], [], [], []
                self.sync_with_global()

        return episode_reward, steps

    def update_global_networks(self, states, actions, rewards, values, dones):
        """Update global networks using collected trajectory"""
        returns, advantages = self.compute_returns(rewards, values, dones)

        # Convert to tensors, processing each state individually due to varying sizes
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Process states one at a time through the networks
        actor_loss = 0
        critic_loss = 0

        for idx in range(len(states)):
            state = torch.FloatTensor(states[idx]).unsqueeze(0)

            # Actor loss
            action_probs = self.local_actor(state)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(actions[idx])
            actor_loss += -(log_prob * advantages[idx].detach())

            # Critic loss
            value_pred = self.local_critic(state).squeeze()
            critic_loss += (value_pred - returns[idx]).pow(2)

        # Average losses
        actor_loss = actor_loss / len(states)
        critic_loss = critic_loss / len(states)

        # Update global networks
        # Actor update
        self.shared_optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), 40)
        for local_param, global_param in zip(
                self.local_actor.parameters(),
                self.global_actor.parameters()):
            if global_param.grad is not None:
                global_param.grad = local_param.grad
        self.shared_optimizer_actor.step()

        # Critic update
        self.shared_optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 40)
        for local_param, global_param in zip(
                self.local_critic.parameters(),
                self.global_critic.parameters()):
            if global_param.grad is not None:
                global_param.grad = local_param.grad
        self.shared_optimizer_critic.step()

    def run(self):
        """Main worker loop"""
        episode_count = 0
        while episode_count < self.max_episodes:
            episode_reward, steps = self.run_episode()
            self.shared_metrics_queue.put((episode_reward, steps))

            # Update shared episode counter
            with self.episode_lock:
                self.shared_episode_counter.value += 1
                current_episode = self.shared_episode_counter.value

            if self.worker_id == 0:
                print(
                    f"\rWorker {self.worker_id} completed episode {current_episode} with reward: {episode_reward:.2f}, "
                    f"steps: {steps}")

            episode_count += 1


class A3CMaster:
    def __init__(self, env_config):
        self.env_config = env_config
        self.global_actor = Actor().share_memory()  # Required for multiprocessing
        self.global_critic = Critic().share_memory()

        # Create shared optimizers
        self.shared_optimizer_actor = torch.optim.Adam(
            self.global_actor.parameters(), lr=1e-4)
        self.shared_optimizer_critic = torch.optim.Adam(
            self.global_critic.parameters(), lr=1e-3)

        # Training metrics
        self.rewards_history = []
        self.steps_history = []

        self.shared_metrics_queue = mp.Queue()

    def train(self, num_episodes, num_workers):
        """Start training with multiple workers"""
        # Create and start workers
        self.shared_episode_counter = mp.Value('i', 0)
        self.episode_lock = mp.Lock()

        # Create and start workers
        workers = []
        episodes_per_worker = num_episodes // num_workers
        for i in range(num_workers):
            worker = A3CWorker(
                i,
                self.global_actor,
                self.global_critic,
                self.env_config,
                self.shared_optimizer_actor,
                self.shared_optimizer_critic,
                self.shared_episode_counter,
                self.episode_lock,
                episodes_per_worker,
                self.shared_metrics_queue
            )
            workers.append(worker)
            worker.start()

        # Monitor progress while workers are running
        while self.shared_episode_counter.value < num_episodes:
            time.sleep(1)
            while not self.shared_metrics_queue.empty():
                reward, steps = self.shared_metrics_queue.get()
                self.rewards_history.append(reward)
                self.steps_history.append(steps)
            print(f"\rProgress: {self.shared_episode_counter.value}/{num_episodes} episodes", end="")
        # Wait for all workers to finish
        for worker in workers:
            worker.join()

        print(f"\nTraining completed: {num_episodes} episodes")

    def evaluate(self, num_episodes):
        """Evaluate the global networks"""
        env = create_flatland_env(**self.env_config)
        rewards = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            state = preprocess_state(obs[0])
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                if info['action_required'][0]:
                    with torch.no_grad():
                        action_probs = self.global_actor(state_tensor)
                    action = torch.argmax(action_probs).item()
                    action_dict = {0: action}
                else:
                    action_dict = {}

                next_obs, reward, dones, info = env.step(action_dict)
                next_state = preprocess_state(next_obs[0])
                done = dones['__all__']

                total_reward += reward[0] if 0 in reward else 0
                state = next_state

            rewards.append(total_reward)

        return np.mean(rewards)

    def save_model(self, path):
        """Save the global networks"""
        torch.save({
            'actor_state_dict': self.global_actor.state_dict(),
            'critic_state_dict': self.global_critic.state_dict(),
        }, path)

    def load_model(self, path):
        """Load the global networks"""
        checkpoint = torch.load(path)
        self.global_actor.load_state_dict(checkpoint['actor_state_dict'])
        self.global_critic.load_state_dict(checkpoint['critic_state_dict'])

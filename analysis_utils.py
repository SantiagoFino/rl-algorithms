import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time


class PerformanceMetrics:
    """
    Class to track and analyze agent performance metrics during training and evaluation.
    """
    current_episode_steps: int = 0
    current_episode_resources: list = []
    episode_start_time: time = time.time()

    def __init__(self):
        self.competitive_rewards_history = []
        self.single_agent_rewards_history = []
        self.resource_collection_times = []
        self.steps_per_episode = []
        self.collection_positions = []

    def reset_episode_metrics(self):
        """Reset metrics for a new episode."""
        self.current_episode_steps = 0
        self.current_episode_resources = []
        self.episode_start_time = time.time()

    def update_episode_metrics(self, resources_collected, step_count, collection_positions):
        """
        Update metrics for current episode.
        Parameters:
            step_count (int): Steps taken in episode
            collection_positions (list): Positions where resources were collected
        """
        self.steps_per_episode.append(step_count)
        self.resource_collection_times.append(time.time() - self.episode_start_time)
        self.collection_positions.extend(collection_positions)

    def update_rewards(self, competitive_reward, single_agent_reward):
        """
        Update reward histories.

        Parameters:
            competitive_reward (float): Reward from competitive agents
            single_agent_reward (float): Reward from single agent
        """
        self.competitive_rewards_history.append(competitive_reward)
        self.single_agent_rewards_history.append(single_agent_reward)

    def plot_learning_curves(self, window_size=100):
        """
        Plot smoothed learning curves for both approaches.

        Parameters:
            window_size (int): Window size for moving average
        """
        plt.figure(figsize=(12, 6))

        # Calculate moving averages
        comp_avg = np.convolve(self.competitive_rewards_history,
                               np.ones(window_size) / window_size,
                               mode='valid')
        single_avg = np.convolve(self.single_agent_rewards_history,
                                 np.ones(window_size) / window_size,
                                 mode='valid')

        plt.plot(comp_avg, label='Competitive MARL')
        plt.plot(single_avg, label='Single Agent RL')
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (over {window_size} episodes)')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_resource_collection_efficiency(self):
        """Plot resource collection efficiency metrics."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(self.steps_per_episode, bins=20)
        plt.xlabel('Steps per Episode')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Lengths')

        plt.subplot(1, 2, 2)
        plt.hist(self.resource_collection_times, bins=20)
        plt.xlabel('Collection Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Resource Collection Time Distribution')

        plt.tight_layout()
        plt.show()

    def plot_collection_heatmap(self, grid_size=5):
        """
        Plot heatmap of resource collection positions.

        Parameters:
            grid_size (int): Size of the environment grid
        """
        collection_grid = np.zeros((grid_size, grid_size))
        for pos in self.collection_positions:
            collection_grid[pos[0], pos[1]] += 1

        plt.figure(figsize=(8, 6))
        sns.heatmap(collection_grid, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Resource Collection Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()


def creates_visualization(env, episode_reward, step):
    clear_output(wait=True)

    # Create grid visualization
    grid = np.full((env.grid_size, env.grid_size), '.')

    # Place agents and resources
    grid[tuple(env.agent1_pos)] = 'A'
    grid[tuple(env.agent2_pos)] = 'B'

    for resource in env.resources:
        if resource is not None:
            grid[tuple(resource)] = 'R'

    # Print grid
    print(f"Step: {step}")
    print(f"Rewards: Agent 1: {episode_reward[0]}, Agent 2: {episode_reward[1]}")
    print("-" * (2 * env.grid_size + 1))
    for row in grid:
        print("|" + " ".join(row) + "|")
    print("-" * (2 * env.grid_size + 1))


def visualize_episode(env, agent1, agent2, max_steps=100, delay=0.5):
    """
    Visualize a single episode of agent interaction.

    Args:
        env: The gymnasium environment
        agent1: First agent
        agent2: Second agent
        max_steps (int): Maximum steps to run
        delay (float): Delay between steps for visualization
    """
    state = env.reset()
    done = False
    episode_reward = [0, 0]
    step = 0

    while not done and step < max_steps:
        # Clear previous output for smooth animation
        creates_visualization(env, episode_reward, step)

        # Get actions
        action1 = agent1.get_action(state, training=False)
        action2 = agent2.get_action(state, training=False)

        # Take step
        state, rewards, done, _ = env.step([action1, action2])
        episode_reward[0] += rewards[0]
        episode_reward[1] += rewards[1]

        step += 1
        time.sleep(delay)

    # Final state
    creates_visualization(env, episode_reward, step)

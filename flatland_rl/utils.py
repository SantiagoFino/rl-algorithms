import numpy as np
import torch
import matplotlib.pyplot as plt

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


def create_flatland_env(width, height, n_agents, max_num_cities):
    """
    Create a Flatland environment with specified parameters.
    """
    rail_generator = sparse_rail_generator(
        max_num_cities=max_num_cities,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=3,
        seed=1
    )

    line_generator = sparse_line_generator(seed=1)

    observation_builder = TreeObsForRailEnv(
        max_depth=2,
        predictor=ShortestPathPredictorForRailEnv(max_depth=10)
    )

    env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        line_generator=line_generator,
        number_of_agents=n_agents,
        obs_builder_object=observation_builder,
        remove_agents_at_target=True
    )

    return env


def preprocess_state(tree_obs):
    """
    Preprocess the tree observation from Flatland environment into a flat vector of fixed size 189.
    """
    if tree_obs is None:
        return np.zeros(189)

    flattened = []

    def _flatten_node(node, depth=0, max_depth=2):
        # Base features per node
        if node is None or depth > max_depth:
            flattened.extend([0] * 9)
            return

        try:
            features = [
                node.dist_own_target_encountered,
                node.dist_other_target_encountered,
                node.dist_other_agent_encountered,
                node.dist_potential_conflict,
                node.dist_unusable_switch,
                node.dist_to_next_branch,
                node.num_agents_same_direction,
                node.num_agents_opposite_direction,
                node.num_agents_malfunctioning
            ]
        except AttributeError:
            features = [0] * 9

        # Normalize features
        normalized = []
        for feat in features:
            if feat == np.inf:
                normalized.append(1.0)
            elif feat == -np.inf:
                normalized.append(-1.0)
            else:
                normalized.append(np.clip(feat / 100, -1, 1))

        flattened.extend(normalized)

        # Process children up to 4 directions
        if depth < max_depth:
            children = getattr(node, 'childs', {})
            for direction in range(4):  # Always process 4 directions
                child = children.get(direction)
                _flatten_node(child, depth + 1, max_depth)

    # Start flattening from root
    _flatten_node(tree_obs)

    # Handle size discrepancy
    if len(flattened) > 189:
        # Truncate if too long
        flattened = flattened[:189]
    elif len(flattened) < 189:
        # Pad with zeros if too short
        flattened.extend([0] * (189 - len(flattened)))

    return np.array(flattened, dtype=np.float32)


def get_tree_state_size():
    """
    Calculate the size of the flattened state vector.
    """
    num_features_per_node = 9
    num_nodes = 1 + 4 + 16
    return num_features_per_node * num_nodes


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        next_value: Value estimate for next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: Computed advantages
        returns: Computed returns
    """
    advantages = []
    returns = []
    gae = 0

    # Add next_value to values list
    values = values + [next_value]

    # Compute GAE
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])

    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def plot_training_results(rewards, steps, window=100):
    """
    Plot training results including rewards and steps.

    Args:
        rewards: List of episode rewards
        steps: List of episode steps
        window: Window size for moving average
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    ax1.plot(np.convolve(rewards, np.ones(window) / window, mode='valid'),
             color='blue', label=f'{window}-Episode Moving Average')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot steps
    ax2.plot(steps, alpha=0.3, color='green', label='Raw Steps')
    ax2.plot(np.convolve(steps, np.ones(window) / window, mode='valid'),
             color='green', label=f'{window}-Episode Moving Average')
    ax2.set_title('Episode Steps')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def update_mean_std_stats(mean_std_tracker, state):
    """
    Update running mean and std statistics for state normalization.

    Args:
        mean_std_tracker: Dictionary containing mean, std, and count
        state: Current state to include in statistics
    """
    if mean_std_tracker['count'] == 0:
        mean_std_tracker['mean'] = state
        mean_std_tracker['std'] = np.zeros_like(state)
        mean_std_tracker['count'] = 1
    else:
        old_mean = mean_std_tracker['mean']
        old_count = mean_std_tracker['count']

        new_count = old_count + 1
        new_mean = old_mean + (state - old_mean) / new_count
        new_std = mean_std_tracker['std'] + (state - old_mean) * (state - new_mean)

        mean_std_tracker['mean'] = new_mean
        mean_std_tracker['std'] = new_std
        mean_std_tracker['count'] = new_count


def normalize_state(state, mean_std_tracker):
    """
    Normalize state using running mean and std.

    Args:
        state: State to normalize
        mean_std_tracker: Dictionary containing mean and std statistics

    Returns:
        normalized_state: Normalized state vector
    """
    if mean_std_tracker['count'] > 1:
        std = np.sqrt(mean_std_tracker['std'] / (mean_std_tracker['count'] - 1))
        std = np.clip(std, 1e-8, np.inf)  # Avoid division by zero
        return (state - mean_std_tracker['mean']) / std
    return state


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Seed value to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Get the device to use for torch operations.

    Returns:
        device: torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_training_state(path, episode, agent, optimizer, rewards, steps):
    """
    Save complete training state.

    Args:
        path: Path to save the checkpoint
        episode: Current episode number
        agent: Agent object containing networks
        optimizer: Optimizer object
        rewards: List of episode rewards
        steps: List of episode steps
    """
    torch.save({
        'episode': episode,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards,
        'steps': steps
    }, path)


def load_training_state(path, agent, optimizer):
    """
    Load complete training state.

    Args:
        path: Path to load the checkpoint from
        agent: Agent object to load networks into
        optimizer: Optimizer object to load state into

    Returns:
        episode: Episode number when checkpoint was saved
        rewards: List of episode rewards
        steps: List of episode steps
    """
    checkpoint = torch.load(path)
    agent.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['rewards'], checkpoint['steps']

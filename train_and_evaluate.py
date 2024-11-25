import numpy as np
import pickle
from datetime import datetime

from rl-algorithms.competitive_envs.gym_env import CompetitiveEnvironment
from rl-algorithms.agents.q_agent import QLearningAgent
from analysis_utils import PerformanceMetrics, visualize_episode


def train_and_evaluate(env: CompetitiveEnvironment, agent1: QLearningAgent, agent2: QLearningAgent,
                       single_agent: QLearningAgent, n_episodes=1000, save_agents=True, metrics=None):
    """
    Train and evaluate both competitive MARL and single-agent RL approaches.

    This function creates an environment and trains:
    1. Two competitive agents learning simultaneously
    2. A single agent learning against a random opponent

    Parameters:
        env (CompetitiveEnvironment) environment where the actions are going to take place
        agent1 (QLearningAgent) Q-learning agent to learn
        agent2 (QLearningAgent) Q-learning agent to learn
        single_agent (QLearningAgent) Q-learning agent to learn
        n_episodes (int) number of episodes to run
        save_agents (bool) whether to save agents trained during training
        metrics (dict) metrics to save during training

    Returns:
        tuple: Average rewards for competitive MARL and single-agent RL
    """
    # Training phase
    competitive_rewards = []
    single_agent_rewards = []

    for episode in range(n_episodes):
        if metrics:
            metrics.reset_episode_metrics()

        # Train competitive agents
        state = env.reset()
        done = False
        episode_rewards = [0, 0]
        step_count = 0
        collection_positions = []
        resources_collected = 0

        while not done:
            actions = [
                agent1.get_action(state),
                agent2.get_action(state)
            ]

            next_state, rewards, done, _ = env.step(actions)

            # Track resource collection positions
            for i, reward in enumerate(rewards):
                if reward > 0:
                    resources_collected += 1
                    collection_positions.append(
                        state[i].tolist()  # Position of agent that collected
                    )

            agent1.learn(state, actions[0], rewards[0], next_state)
            agent2.learn(state, actions[1], rewards[1], next_state)

            state = next_state
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            step_count += 1

        competitive_rewards.append(sum(episode_rewards) / 2)

        if metrics:
            metrics.update_episode_metrics(
                resources_collected=resources_collected,
                step_count=step_count,
                collection_positions=collection_positions
            )

        # Train single agent against random opponent
        state = env.reset()
        done = False
        episode_rewards = [0, 0]
        step_count = 0
        collection_positions = []
        resources_collected = 0

        while not done:
            action1 = single_agent.get_action(state)
            action2 = env.action_space.sample()  # Random opponent

            next_state, rewards, done, _ = env.step([action1, action2])

            # Track resource collection
            if rewards[0] > 0:
                resources_collected += 1
                collection_positions.append(state[0].tolist())

            single_agent.learn(state, action1, rewards[0], next_state)

            state = next_state
            episode_rewards[0] += rewards[0]
            step_count += 1

        single_agent_rewards.append(episode_rewards[0])

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            comp_avg = np.mean(competitive_rewards[-100:])
            single_avg = np.mean(single_agent_rewards[-100:])
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Competitive MARL avg reward (last 100): {comp_avg:.2f}")
            print(f"Single Agent RL avg reward (last 100): {single_avg:.2f}")
            print("-" * 50)

        if metrics:
            metrics.update_rewards(competitive_rewards[-1], single_agent_rewards[-1])

    # Calculate final averages
    competitive_avg = np.mean(competitive_rewards[-100:])
    single_agent_avg = np.mean(single_agent_rewards[-100:])

    # Save trained agents
    if save_agents:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"trained_agents_{timestamp}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump({
                'agent1': agent1,
                'agent2': agent2,
                'single_agent': single_agent,
                'training_history': {
                    'competitive_rewards': competitive_rewards,
                    'single_agent_rewards': single_agent_rewards
                }
            }, f)
        print(f"Saved trained agents to: {save_path}")

    return competitive_avg, single_agent_avg


def load_trained_agents(filepath):
    """
    Load previously trained agents from a file.

    Args:
        filepath (str): Path to the saved agents file

    Returns:
        tuple: (agent1, agent2, single_agent, training_history)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return (
        data['agent1'],
        data['agent2'],
        data['single_agent'],
        data['training_history']
    )


if __name__ == "__main__":
    environment = CompetitiveEnvironment()
    competitive_reward, single_agent_reward = train_and_evaluate(env=environment,
                                                                 agent1=QLearningAgent(environment.action_space),
                                                                 agent2=QLearningAgent(environment.action_space),
                                                                 single_agent=QLearningAgent(environment.action_space))
    print(f"Average reward (last 100 episodes):")
    print(f"Competitive MARL: {competitive_reward:.2f}")
    print(f"Single Agent RL: {single_agent_reward:.2f}")
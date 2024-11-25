from random import random
from collections import defaultdict
import numpy as np
import gymnasium as gym


class QLearningAgent:
    """
        Implementation of a Q-Learning agent.

        This agent learns to make decisions through Q-learning, a model-free
        reinforcement learning algorithm that learns an action-value function
        representing the expected utility of taking a given action in a given state.

        Attributes:
            action_space (gym.spaces.Discrete): The action space of the environment
            lr (float): Learning rate for Q-value updates
            gamma (float): Discount factor for future rewards
            epsilon (float): Exploration rate for epsilon-greedy policy
            q_table (defaultdict): Table storing Q-values for state-action pairs
        """
    def __init__(self, action_space: gym.spaces.Discrete, learning_rate: float=0.1, discount_factor: float=0.95,
                 epsilon: float=0.1):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def get_action(self, state: np.ndarray, training: bool=True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters:
            state (numpy.ndarray): Current observation of the environment
            training (bool): Whether the agent is training (affects exploration)

        Returns:
            int: Selected action
        """
        state = tuple(map(tuple, state))

        if training and random() < self.epsilon:
            return int(self.action_space.sample())

        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        """
        Update Q-values based on the observed transition.
    
        Args:
            state (numpy.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Resulting state
        """
        # Convert numpy arrays to tuples for dictionary key
        state_tuple = tuple(map(tuple, state))
        next_state_tuple = tuple(map(tuple, next_state))
        
        old_value = self.q_table[state_tuple][action]
        next_max = np.max(self.q_table[next_state_tuple])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state_tuple][action] = new_value

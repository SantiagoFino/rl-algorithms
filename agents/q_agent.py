from random import random
from collections import defaultdict
import numpy as np
import gymnasium as gym


class QLearningAgent:
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
        """
        # Convert state to a hashable tuple
        state_tuple = tuple(map(lambda x: tuple(x), state))

        if training and random() < self.epsilon:
            return int(self.action_space.sample())

        return int(np.argmax(self.q_table[state_tuple]))

    def learn(self, state, action, reward, next_state):
        """
        Update Q-values based on the observed transition.
        """
        # Convert states to hashable tuples
        state_tuple = tuple(map(lambda x: tuple(x), state))
        next_state_tuple = tuple(map(lambda x: tuple(x), next_state))

        # Get current Q-value
        old_value = self.q_table[state_tuple][action]
        
        # Get maximum Q-value for next state
        next_max = np.max(self.q_table[next_state_tuple])

        # Update Q-value using the Q-learning formula
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state_tuple][action] = new_value

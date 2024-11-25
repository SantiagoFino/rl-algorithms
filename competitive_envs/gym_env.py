import numpy as np
import gymnasium as gym
import random


class CompetitiveEnvironment(gym.Env):
    """
    A competitive environment where two agents compete for limited resources on a grid.

    This environment implements a simple grid world where two agents move around
    trying to collect resources. Agents can block each other's paths and must
    develop strategies to maximize their resource collection.

    Attributes:
        grid_size (int): Size of the square grid (grid_size x grid_size)
        n_resources (int): Number of resources available in the environment
        max_steps (int): Maximum number of steps before episode termination
        action_space (gym.spaces.Discrete): Discrete action space for movement 0: up, 1: right, 2: down, 3: left
        observation_space (gym.spaces.Box): Box space for agent and resource positions
    """
    action_space: gym.spaces = gym.spaces.Discrete(4)
    step_count: int = 0

    def __init__(self, grid_size: int = 10, n_resources: int = 3, max_steps: int = 100):
        self.grid_size = grid_size
        self.n_resources = n_resources
        self.max_steps = max_steps

        # describes how one observation of the agent would look like
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2 + self.n_resources, 2), dtype=np.int32
        )
        self.agent1_pos, self.agent2_pos = np.array([0, 0]), np.array([0, 0])  # default values for the agents positions
        self.resources: list[np.array] = [np.array([0, 0]) for _ in range(self.n_resources)]  # default values for the res positions

    def _get_obs(self) -> np.ndarray:
        """
        Gets the current observation of the environment.

        Returns:
           numpy.ndarray: Array containing positions of agents and resources
        """
        obs = np.zeros((2 + self.n_resources, 2), dtype=np.int32)
        obs[0] = self.agent1_pos
        obs[1] = self.agent2_pos

        for i, resource in enumerate(self.resources):
            if resource is not None:
                obs[i+2] = resource
        return obs

    def _move_agents(self, moves: dict[int, list], actions: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Moves a given agent based on the taken action, the current agent position and the available moves dictionary.

        Handles collision between agents by randomly selecting which agent gets to move to a contested position while
        the other stays in place.

        Parameters:
            moves (dict[int, list]): Dictionary of agents and their moves
            actions (list[int]): Action taken by the agent

        Returns:
            np.array: New position of the agent
        """
        new_pos1 = self.agent1_pos + moves[actions[0]]
        new_pos2 = self.agent2_pos + moves[actions[1]]

        # Check if new positions are within grid bounds
        pos1_valid = (0 <= new_pos1[0] < self.grid_size and
                      0 <= new_pos1[1] < self.grid_size)
        pos2_valid = (0 <= new_pos2[0] < self.grid_size and
                      0 <= new_pos2[1] < self.grid_size)

        # Handle collisions
        if pos1_valid and pos2_valid and np.array_equal(new_pos1, new_pos2):
            if random.random() < 0.5:  # Agent 1 gets the spot and agent 2 stays in place
                self.agent1_pos = new_pos1
            else:  # Agent 2 gets the spot and agent 1 stays in place
                self.agent2_pos = new_pos2
        else:
            if pos1_valid:
                # Also check if agent 2 is not already at the new position
                if not np.array_equal(new_pos1, self.agent2_pos):
                    self.agent1_pos = new_pos1

            if pos2_valid:
                # Also check if agent 1 is not already at the new position
                if not np.array_equal(new_pos2, self.agent1_pos):
                    self.agent2_pos = new_pos2

        return self.agent1_pos, self.agent2_pos

    def _collect_rewards(self) -> list[int]:
        """
        Collects reward of each agent on the grid
        Returns:
            list[int]: List with the collected rewards by each agent
        """
        rewards = [0, 0]
        for i, collected_resource in enumerate(self.resources):
            if collected_resource is not None:
                if np.array_equal(self.agent1_pos, collected_resource):
                    rewards[0] += 1
                # elif because the agents cannot be in the same position
                elif np.array_equal(self.agent2_pos, collected_resource):
                    rewards[1] += 1

        # remove the resources
        for i, resource in enumerate(self.resources):
            if resource is not None:
                if np.array_equal(self.agent1_pos, resource) or np.array_equal(self.agent2_pos, resource):
                    self.resources[i] = None

        return rewards

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state by randomly placing agents and resources on different grid positions.

        Returns:
            numpy.ndarray: Initial observation of the environment
        """
        self.step_count = 0

        positions = np.random.choice(
            self.grid_size ** 2,
            size=2 + self.n_resources,  # number of agents + the amount of resources
            replace=False
        )

        self.agent1_pos = np.array([positions[0] // self.grid_size,
                                    positions[0] % self.grid_size])
        self.agent2_pos = np.array([positions[1] // self.grid_size,
                                    positions[1] % self.grid_size])

        # initialize resources
        self.resources = []
        for i in range(self.n_resources):
            self.resources.append(
                np.array([positions[i + 2] // self.grid_size,  # positions of the resources in the grid in coordinates
                          positions[i + 2] % self.grid_size])  # format array([i, j])
            )

        return self._get_obs()

    def step(self, actions: list[int]) -> tuple:
        """
        Take a step in the environment given the actions of both agents.

        Parameters:
            actions (list): List containing actions for both agents [agent1_action, agent2_action]

        Returns:
            tuple: with the observations, the rewards, the done flag and an info dict with any additional information
            about the step
        """
        self.step_count += 1

        possible_moves = {
            0: [-1, 0],  # up
            1: [0, 1],  # right
            2: [1, 0],  # down
            3: [0, -1]  # left
        }

        # moves the agents
        self.agent1_pos, self.agent2_pos = self._move_agents(possible_moves, actions)

        # checks the resource collection
        rewards = self._collect_rewards()
        
        # checks if the game is over if the game surpasses the max amount of steps or all the resources are collected
        done = (self.step_count >= self.max_steps or all(resource is None for resource in self.resources))

        return self._get_obs(), rewards, done, {}

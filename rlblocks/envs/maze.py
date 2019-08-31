import gym
from gym.spaces import Discrete
import numpy as np
import random
from gridgym.envs.envbase import BaseEnv


class MazeEnv(BaseEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        # action-space
        self._action_set = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        self._action_meaning = ["^", "<", "v", ">"]
        self.action_space = Discrete(len(self._action_set))

        self.grid_size = 39
        self.states_count = self.grid_size ** 2
        self.observation_space = Discrete(self.states_count)

        self.grid = self._generate_complex_grid(
            width=self.grid_size, height=self.grid_size)

        # TODO: check if position is available to use in complex grid
        self.position = [1, 1]
        self.reset()

    # SOURCE: https://gist.github.com/rougier/1d0e9f56e8bb4d654ed73062fb0a8766
    def _generate_complex_grid(self, width, height, complexity=.05, density=.1):

        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        # number of components
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        # size of components
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        grid = np.zeros(shape, dtype=int)
        # Fill borders
        grid[0, :] = grid[-1, :] = self.WALL_TILE
        grid[:, 0] = grid[:, -1] = self.WALL_TILE
        # Make aisles
        for i in range(density):
            # pick a random position
            x, y = random.randint(
                0, shape[1] // 2) * 2, random.randint(0, shape[0] // 2) * 2
            grid[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                    if grid[y_, x_] == 0:
                        grid[y_, x_] = self.WALL_TILE
                        grid[y_ + (y - y_) // 2, x_ + (x - x_) //
                             2] = self.WALL_TILE
                        x, y = x_, y_
        return grid

    def render(self, mode='human'):
        pass

    def reset(self, start_state=None):

        if start_state is None:
            self.position = [1, 1]
        else:
            self.position = self._state_to_position(start_state)

        if self.grid[self.position[0]][self.position[1]] == self.WALL_TILE:
            raise ValueError('starting position is a non-valid position')

        return self._position_to_state(self.position)

    def step(self, action):

        action_pos = self._action_set[action]
        new_position = np.add(self.position, action_pos)

        try:
            if self.grid[new_position[0]][new_position[1]] == self.FREE_TILE:
                self.position = new_position
        except:
            # position out or range
            self.position = self.position

        return self._position_to_state(self.position), 0, False, {}

import gym
import random
import math


class BaseEnv(gym.Env):

    WALL_TILE = 1
    FREE_TILE = 0

    def get_free_rand_state(self):
        while True:
            state = random.randint(0, self.states_count - 1)
            position = self._state_to_position(state)

            if self.grid[position[0]][position[1]] != self.WALL_TILE:
                return state

    def _state_to_position(self, state):
        row = math.floor(state / self.grid_size)
        col = state % self.grid_size
        return row, col

    def _position_to_state(self, position):
        row = position[0]
        col = position[1]

        return row * self.grid_size + col

from collections import deque
from random import choices
from rlblocks.memory.memory_interface import MemoryBase
import numpy as np

class ExperienceReplayMemory(MemoryBase):

    def __init__(self, capacity=int(1e6), device="cpu"):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def store(self, transition):
        self.data.append(transition)

    def sample(self, size):
        if len(self.data) <= size:
            return self.data

        return choices(self.data, k=size)

    def save(self, location):
        # loction is the folder where the different files should be stored

        batch = zip(*self.data)

        for index in range(len(self.data[0])):
            col_data = np.array(next(batch))
            np.savez_compressed(f"{location}/mem_{index}", col_data)

    def load(self, filename):
        self.data = np.load(filename)

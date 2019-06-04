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

    def save(self, filename="memory.npz"):
        np.savez_compressed(filename, self.data)

    def load(self, filename):
        self.data = np.load(filename)

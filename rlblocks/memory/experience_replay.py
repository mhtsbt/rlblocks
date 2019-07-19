from collections import deque
from random import choices
from rlblocks.memory.memory_interface import MemoryBase
import numpy as np
from glob import glob

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

    def load(self, location):

        files = glob(f"{location}/mem_*.npz")

        loaded_data = []

        for file in files:

            file_content = np.load(file)
            file_data = file_content['arr_0']
            loaded_data.append(file_data)

        n_columns = len(loaded_data)

        # very inefficient way to load the data
        for row in range(len(loaded_data[0])):
            t = []
            for col in range(n_columns):
                t.append(loaded_data[col][row])

            self.store(t[::-1])

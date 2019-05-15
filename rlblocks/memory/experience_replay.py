from collections import deque
from random import choices


class ExperienceReplayMemory:

    def __init__(self, capacity=int(100e3)):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def store(self, transition):
        self.data.append(transition)

    def sample(self, size):
        if len(self.data) <= size:
            return self.data

        return choices(self.data, k=size)

    def save(self):
        return
        # TODO
        #print("Saving memory")
        #np.save("memory.npy", self.data)

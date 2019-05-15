from random import random
from torch import FloatTensor


class EpsGreedyPolicy:

    def __init__(self, model, action_space, device):
        self.action_space = action_space
        self.device = device
        self.model = model

    def sample_action(self, state, eps):

        if random() < eps:
            action = self.action_space.sample()
        else:
            state_tensor = FloatTensor([state]).to(self.device)
            _, action = self.model(state_tensor).max(1)
            action = action.item()

        return action

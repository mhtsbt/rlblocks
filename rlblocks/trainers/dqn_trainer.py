from torch.optim import Adam, RMSprop
from torch import FloatTensor, LongTensor
from torch.nn.functional import smooth_l1_loss
import numpy as np


class DQNNatureTrainer:

    def __init__(self, model, target_model, optimizer, gamma=0.99, device='cuda'):

        self.model = model
        self.target_model = target_model

        self.device = device
        self.gamma = gamma
        self.optimizer = optimizer

    def train(self, start_states, result_states, actions, rewards, done):

        start_state = FloatTensor(np.array(start_states)).to(self.device)
        result_state = FloatTensor(np.array(result_states)).to(self.device)
        actions = LongTensor(np.array(actions)).to(self.device).unsqueeze(1)
        rewards = FloatTensor(np.array(rewards)).to(self.device)

        # future values
        future_values, _ = self.target_model(result_state).max(1)
        future_values = future_values.detach()

        # set future value to zero for final states
        nonterminal = FloatTensor(np.invert(done)).to(self.device)

        current_values = self.model(start_state)
        current_values = current_values.gather(1, actions)

        target = rewards + nonterminal * self.gamma * future_values
        target = target.unsqueeze(1)

        # Compute Huber loss
        loss = smooth_l1_loss(current_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

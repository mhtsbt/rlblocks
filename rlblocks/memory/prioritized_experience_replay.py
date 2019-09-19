from collections import namedtuple
import numpy as np
import torch
import os
from rlblocks.utils.segment_tree import SegmentTree
from config import Config
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal', 'keys'))


class PrioritizedExperienceReplayMemory:

    def __init__(self, device='cpu', priority_exponent=0.5, memory_capacity=int(1e6), discount=0.99, frame_width=84, frame_height=84, frame_history=1, multi_step=3):

        self.device = device
        self.priority_weight = 0.4
        self.priority_exponent = priority_exponent # TODO: this was removed in dopamine
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_history = frame_history
        self.multi_step = multi_step
        self.memory_capacity = memory_capacity
        self.discount = discount

        self.t = 0

        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(memory_capacity)

        self.blank_trans = Transition(0, torch.zeros(1, self.frame_height, self.frame_width, dtype=torch.uint8), None, 0, False, 0)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal, keys):

        # Only store last frame
        state_tensor = torch.ByteTensor([np.array(state[-1])]).to(torch.device('cpu'))

        # Store new transition with maximum priority
        self.transitions.append(Transition(self.t, state_tensor, action, reward, not terminal, keys), self.transitions.max)

        # Start new episodes with t = 0
        self.t = 0 if terminal else self.t + 1

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):

        transition = np.array([None] * (self.frame_history + self.multi_step))
        transition[self.frame_history - 1] = self.transitions.get(idx)

        # get the previously stored frames
        for t in range(self.frame_history - 2, -1, -1):  # e.g. 2 1 0 with history 4
            if transition[t + 1].timestep == 0:
                # If future frame has timestep 0
                transition[t] = self.blank_trans
            else:
                index = idx + (-self.frame_history + t + 1)
                transition[t] = self.transitions.get(index)

        # get future frames to calculate the n-step return
        for t in range(self.frame_history, self.frame_history + self.multi_step):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                index = idx + (-self.frame_history + t + 1)
                transition[t] = self.transitions.get(index)

                if not transition[t]:
                    # transition could not be found, fill with empty trans
                    transition[t] = self.blank_trans

            else:
                transition[t] = self.blank_trans  # If prev (next) frame is terminal

        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:

            # Uniformly sample an element from within a segment
            sample = np.random.uniform(i * segment, (i + 1) * segment)

            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = self.transitions.find(sample)

            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.memory_capacity > self.multi_step and (idx - self.transitions.index) % self.memory_capacity >= self.frame_history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)

        # Create un-discretised state and nth next state
        state = (torch.stack([trans.state for trans in transition[:self.frame_history]]).to(dtype=torch.float32, device=self.device).div_(127.5)-1.0)[0]

        test = [trans.state for trans in transition[self.multi_step:self.multi_step + self.frame_history]]
        next_state = (torch.stack(test).to(dtype=torch.float32, device=self.device).div_(127.5)-1.0)[0]

        # Discrete action to be used as index
        action = torch.tensor([transition[self.frame_history - 1].action], dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.frame_history + n - 1].reward for n in range(self.multi_step))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.frame_history + self.multi_step - 1].nonterminal], dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):

        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)

        if np.isnan(p_total):
            p_total = 0

        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities

        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.cat(actions).to(self.device)
        returns = torch.cat(returns).to(self.device)
        nonterminals = torch.stack(nonterminals).to(self.device)

        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.memory_capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def save(self, filename):

        data_dir = os.path.join(".", "memory")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print("Saving exp memory")

        # TODO: also store priority?
        mem_savable = list(map(lambda exp: (exp.timestep, exp.state.data.numpy(), exp.action, exp.reward, exp.nonterminal), self.transitions.data[:self.transitions.index]))
        np.save(os.path.join(data_dir, filename), mem_savable)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.memory_capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.frame_history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.frame_history - 1)):
            if prev_timestep == 0:
                state_stack[t] = self.blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.frame_history + 1].state
                prev_timestep -= 1
        state = (torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(127.5)-1.0)[0]  # Agent will turn into batch
        self.current_idx += 1
        return state

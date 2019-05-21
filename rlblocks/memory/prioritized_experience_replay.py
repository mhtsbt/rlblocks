from collections import namedtuple
import numpy as np
import torch
import os
from rlblocks.utils.segment_tree import SegmentTree
from rlblocks.memory.memory_interface import MemoryBase

# TODO: remove this transition
Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))

#Transition = namedtuple('Transition', ['start_state', 'result_state', 'action', 'reward', 'done', 'fs'])


class PrioritizedExperienceReplayMemory(MemoryBase):

    def __init__(self, capacity=int(1e6), device="cuda"):

        self.device = device
        self.capacity = capacity
        self.priority_exponent = self.config.priority_exponent # TODO: this was removed in dopamine

        # Internal episode timestep counter
        self.t = 0

        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(self.config.memory_capacity)

        self.blank_trans = Transition(0, torch.zeros((self.config.frame_height, self.config.frame_width), dtype=torch.uint8), None, 0, False)

    def store(self, transition):
        # Only store last frame
        transition.start_state = torch.ByteTensor(np.array(transition.start_state[-1])).to(torch.device('cpu'))

        # we do not need the result state because we use indexes
        transition.result_state = None

        # Store new transition with maximum priority
        transition.timestep = self.t
        self.transitions.append(transition, self.transitions.max)

        # update the frame-index
        if transition.done:
            self.t = 0
        else:
            self.t += 1

    def _get_transition(self, idx, frame_stack=4, multi_step=3):

        # build the basic structure
        transition = np.array([None] * (frame_stack + multi_step))
        transition[frame_stack - 1] = self.transitions.get(idx)

        # get the frame history
        for t in range(frame_stack - 2, -1, -1):  # e.g. 2 1 0 with history 4
            if transition[t + 1].timestep == 0:
                # If future frame has timestep 0
                transition[t] = self.blank_trans
            else:
                index = idx + (-self.frame_stack + t + 1)
                transition[t] = self.transitions.get(index)

        # get future frames to calculate the n-step return
        for t in range(frame_stack, frame_stack + multi_step):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                index = idx + (-frame_stack + t + 1)
                transition[t] = self.transitions.get(index)

                if not transition[t]:
                    # transition could not be found, fill with empty trans
                    transition[t] = self.blank_trans
            else:
                # If prev (next) frame is terminal
                transition[t] = self.blank_trans

        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i, frame_stack=4, multi_step=3, discount=0.99):

        # TODO: convertion to float is an implementation detail so should probably also be used in append method

        valid = False

        while not valid:
            # Uniformly sample an element from within a segment
            sample = np.random.uniform(i * segment, (i + 1) * segment)

            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = self.transitions.find(sample)

            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > multi_step and (idx - self.transitions.index) % self.capacity >= frame_stack and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)

        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:frame_stack]]).to(dtype=torch.float32, device=self.device).div_(255)

        test = [trans.state for trans in transition[multi_step:multi_step + frame_stack]]
        next_state = torch.stack(test).to(dtype=torch.float32, device=self.device).div_(255)

        # Discrete action to be used as index
        action = torch.tensor([transition[frame_stack - 1].action], dtype=torch.int64, device=self.device)

        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(discount ** n * transition[frame_stack + n - 1].reward for n in range(multi_step))], dtype=torch.float32, device=self.device)

        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([not transition[frame_stack + multi_step - 1].done], dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, size):
        # Retrieve sum of all priorities (used to create a normalised probability distribution)
        p_total = self.transitions.total()

        # Batch size number of segments, based on sum over all probabilities
        segment = p_total / size

        # Get batch of valid samples
        batch = [self._get_sample_from_segment(segment, i) for i in range(size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.cat(actions).to(self.device)
        returns = torch.cat(returns).to(self.device)
        nonterminals = torch.stack(nonterminals).to(self.device)

        # Calculate normalised probabilities
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index

        # Compute importance-sampling weights w
        weights = (capacity * probs) ** -self.config.priority_weight

        # Normalise by max importance-sampling weight from batch
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):

        # TODO

        priorities = np.power(priorities, self.config.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def save(self, filename):

        data_dir = os.path.join(self.config.data_path, "memory")

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
        if self.current_idx == self.config.memory_capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.config.frame_history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.config.frame_history - 1)):
            if prev_timestep == 0:
                state_stack[t] = self.blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.config.frame_history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.config.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state

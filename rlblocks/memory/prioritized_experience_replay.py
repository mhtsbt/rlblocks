from collections import namedtuple
import numpy as np
import torch
import os
from rlblocks.utils.segment_tree import SegmentTree

# TODO: remove this transition
Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


class PrioritizedExperienceReplayMemory:

    def __init__(self, config: Config):

        self.config = config
        self.priority_exponent = self.config.priority_exponent # TODO: this was removed in dopamine

        # Internal episode timestep counter (for each worker)
        self.t = np.zeros(shape=self.config.n_workers, dtype=np.int)

        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(self.config.memory_capacity)

        self.blank_trans = Transition(0, torch.zeros((self.config.frame_height, self.config.frame_width), dtype=torch.uint8), None, 0, False)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal, worker_id):

        # Only store last frame
        state_tensor = torch.ByteTensor(np.array(state[-1])).to(torch.device('cpu'))

        # Store new transition with maximum priority
        self.transitions.append(Transition(self.t[worker_id], state_tensor, action, reward, not terminal), self.transitions.max)

        # Start new episodes with t = 0
        self.t[worker_id] = 0 if terminal else self.t[worker_id] + 1

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):

        transition = np.array([None] * (self.config.frame_history + self.config.multi_step))
        transition[self.config.frame_history - 1] = self.transitions.get(idx)

        # get the previously stored frames
        for t in range(self.config.frame_history - 2, -1, -1):  # e.g. 2 1 0 with history 4
            if transition[t + 1].timestep == 0:
                # If future frame has timestep 0
                transition[t] = self.blank_trans
            else:
                index = idx + (-self.config.frame_history + t + 1)*self.config.n_workers
                transition[t] = self.transitions.get(index)

        # get future frames to calculate the n-step return
        for t in range(self.config.frame_history, self.config.frame_history + self.config.multi_step):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                index = idx + (-self.config.frame_history + t + 1)*self.config.n_workers
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

            # TODO: use worker_id to determine if a sample is valid or not

            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.config.memory_capacity > (self.config.multi_step * self.config.n_workers) and (idx - self.transitions.index) % self.config.memory_capacity >= (self.config.frame_history*self.config.n_workers) and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)

        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.config.frame_history]]).to(dtype=torch.float32, device=self.config.device).div_(255)

        test = [trans.state for trans in transition[self.config.multi_step:self.config.multi_step + self.config.frame_history]]
        next_state = torch.stack(test).to(dtype=torch.float32, device=self.config.device).div_(255)

        # Discrete action to be used as index
        action = torch.tensor([transition[self.config.frame_history - 1].action], dtype=torch.int64, device=self.config.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.config.discount ** n * transition[self.config.frame_history + n - 1].reward for n in range(self.config.multi_step))], dtype=torch.float32, device=self.config.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.config.frame_history + self.config.multi_step - 1].nonterminal], dtype=torch.float32, device=self.config.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):

        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities

        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.cat(actions).to(self.config.device)
        returns = torch.cat(returns).to(self.config.device)
        nonterminals = torch.stack(nonterminals).to(self.config.device)

        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.config.memory_capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.config.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.config.device)  # Normalise by max importance-sampling weight from batch

        #for i in range(10):
        #    test = states.numpy()[i]
        #    plt.imshow(np.concatenate(test))
        #    plt.show()

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
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

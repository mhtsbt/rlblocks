import numpy as np
from torch.optim import Adam
import torch


class RainbowTrainer:

    def __init__(self, config, logger, model, target_model, supports, memory):
        self.config = config
        self.logger = logger
        self.model = model
        self.target_model = target_model
        self.supports = supports
        self.memory = memory
        self.loss_history = []

        self.optimizer = Adam(self.model.parameters(), lr=self.config.adam_lr, eps=self.config.adam_eps)

        # keep track of the training-step (over all iterations), this is used for logging the loss
        self.training_step = 0

    @staticmethod
    def _get_cross_entropy_loss(current_dist, target_dist):
        loss = (-target_dist * current_dist.log()).sum(-1)
        return loss

    def _get_current_dist_from_sample(self, states, actions):
        current_dist = self.model(states)
        # only work with the distribution for the sampled action
        current_dist = current_dist.gather(1, actions)

        return current_dist.squeeze(1)

    def _get_target_dist_for_sample(self, states, result_states, actions, rewards, nonterminals):

        with torch.no_grad():

            # Calculate nth next state probabilities
            pns = self.target_model(result_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.supports.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)

            # collect the value of the action-distribution with the highest value
            pns_a = pns[range(self.config.train_batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            tz = rewards.unsqueeze(1) + nonterminals * (self.config.discount ** self.config.multi_step) * self.supports.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)

            # Clamp between supported values
            tz = tz.clamp(min=self.config.v_min, max=self.config.v_max)

            # Compute L2 projection of Tz onto fixed support z
            b = (tz - self.config.v_min) / self.config.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.config.train_batch_size, self.config.atoms)
            offset = torch.linspace(0, ((self.config.train_batch_size - 1) * self.config.atoms), self.config.train_batch_size).unsqueeze(1).expand(self.config.train_batch_size, self.config.atoms).to(actions)

            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))

            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        return m

    def _get_sample_loss(self, sample):

        idxs, states, actions, rewards, result_states, nonterminals, weights = sample

        # extend actions to [batch x 1 x atoms]
        extended_actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.config.atoms)

        # get current and target distributions in order to calculate the loss
        current_dist = self._get_current_dist_from_sample(states=states, actions=extended_actions)
        target_dist = self._get_target_dist_for_sample(states=states, result_states=result_states, actions=extended_actions, rewards=rewards, nonterminals=nonterminals)

        # calculate the cross-entropy loss KL(m||p(s_t,a_t))
        loss = self._get_cross_entropy_loss(current_dist=current_dist, target_dist=target_dist)

        return loss

    def do_training_step(self, sample):

        idxs, states, actions, rewards, result_states, nonterminals, weights = sample

        loss = self._get_sample_loss(sample)
        weighted_loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        self.loss_history.append(weighted_loss.item())

        # log training process every x updates
        if self.training_step % 1000 == 0:
            # use the average of the last x training steps
            avg_loss = np.average(self.loss_history)
            self.logger.scalar("train/loss", avg_loss, self.training_step)
            self.loss_history = []

        # PER: update priorities of sampled transitions
        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())

        # increment training step, to facilitate logging loss
        self.training_step += 1

        return weighted_loss.item()

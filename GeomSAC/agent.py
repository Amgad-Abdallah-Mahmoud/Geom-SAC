import torch.optim as optim

from buffer import *
from neural_networks import *


class SoftActorCriticAgent:
    def __init__(self, env, state):
        torch.autograd.set_detect_anomaly(True)
        self.env = env
        self.n_actions = sum(env.action_space.nvec)
        self.action_space = env.action_space.nvec.tolist()
        self.ac_loss, self.q_loss = [], []

        self.critic_v = StateValueNetwork(state)
        self.critic_v_target = StateValueNetwork(state)
        self.critic_q_1 = ActionValueNetwork(state, self.n_actions)
        self.critic_q_2 = ActionValueNetwork(state, self.n_actions)
        self.actor = PolicyNetwork(env, state, self.n_actions)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3 * 10e-4)  # 0.003
        self.v_optim = optim.Adam(self.critic_v.parameters(), lr=0.003)
        self.q1_optim = optim.Adam(self.critic_q_1.parameters(), lr=0.003)
        self.q2_optim = optim.Adam(self.critic_q_2.parameters(), lr=0.003)

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 32
        self.reward_scale = 10
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.update_target(1)

    def select_actions(self, state, reparam=True):
        self.actor.train()
        with torch.no_grad():
            probabilities, actions, log_p = self.actor.sample(state, self.action_space)
        return probabilities, actions, log_p

    def train(self):
        if len(self.replay_buffer.replay_buffer) < self.batch_size:
            return

        samples = self.replay_buffer.sample()
        states = samples[0]
        probabilities = samples[1]
        rewards = samples[2]
        next_states = samples[3]
        dones = samples[4]

        current_q_1 = self.critic_q_1(states, probabilities)
        current_q_2 = self.critic_q_2(states, probabilities)
        current_critic_v = self.critic_v(states)

        _, _, log_p = self.actor.sample(states, self.env.action_space.nvec.tolist())

        target_q = rewards * self.reward_scale + (self.gamma * self.critic_v_target(next_states) * (1 - dones))

        q1_loss = F.mse_loss(current_q_1, target_q.detach())
        q2_loss = F.mse_loss(current_q_2, target_q.detach())
        self.q_loss.append((q1_loss, q2_loss))

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        q1 = self.critic_q_1(states, probabilities)
        q2 = self.critic_q_2(states, probabilities)

        predicted_new_q = torch.min(q1, q2)
        target_critic_v = predicted_new_q - log_p.unsqueeze(1)
        critic_loss = F.mse_loss(current_critic_v, target_critic_v.detach())
        self.v_optim.zero_grad()
        critic_loss.backward()
        self.v_optim.step()

        actor_loss = (log_p * (log_p - predicted_new_q)).mean()
        self.ac_loss.append(actor_loss)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.update_target(self.tau)

    def update_target(self, tau):
        for target_param, param in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

import os

import torch
import torch.nn.functional as F
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import GAT, GIN


class GraphEncoder(nn.Module):
    def __init__(self, state, n_layers=1, dim_h=128, heads=4):
        super().__init__()
        num_node_features = state.num_node_features
        self.conv1 = GAT(num_node_features, dim_h, n_layers, v2=True, heads=heads)
        self.conv2 = GIN(dim_h, dim_h, n_layers)

        self.fc1 = nn.Linear(dim_h, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_h)

    def forward(self, state, batch=None):
        x, edge_index = state.x, state.edge_index
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)

        h = global_add_pool(h2, batch)

        x = self.fc1(h)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x


class StateValueNetwork(nn.Module):
    def __init__(self, states, dim_h=128, chkpt_dir='tmp/GeomSac'):
        super().__init__()
        self.fc1 = nn.Linear(dim_h, dim_h)
        self.fc2 = nn.Linear(dim_h, 64)
        self.fc3 = nn.Linear(64, 1)
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, 'value_geom_sac')

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActionValueNetwork(nn.Module):
    def __init__(self, states, n_actions, dim_h=128, chkpt_dir='tmp/GeomSac'):
        super().__init__()
        self.fc1 = nn.Linear(dim_h + n_actions, dim_h)
        self.fc2 = nn.Linear(dim_h, 64)
        self.fc3 = nn.Linear(64, 1)
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, 'critic_geom_sac')

    def forward(self, state, action):
        x = self.fc1(torch.cat((state, action), dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class PolicyNetwork(nn.Module):
    def __init__(self, env, states, n_actions, dim_h=128, chkpt_dir='tmp/GeomSac'):
        super().__init__()
        self.fc1 = nn.Linear(dim_h, dim_h)
        self.fc2 = nn.Linear(dim_h, n_actions)

        self.sm_ac1 = nn.Softmax(dim=-1)
        self.sm_ac2 = nn.Softmax(dim=-1)
        self.sm_ac3 = nn.Softmax(dim=-1)
        self.sm_ac4 = nn.Softmax(dim=-1)
        self.sm_ac5 = nn.Softmax(dim=-1)

        self.md = MultiCategoricalDistribution(env.action_space.nvec.tolist())
        self.probs_fc = self.md.proba_distribution_net(n_actions)

        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'actor_sac')

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.probs_fc(x)
        return logits

    def sample(self, state, action_space, epsilon=1e-6):
        logits = self.forward(state)
        ac1_probs = self.sm_ac1(logits[0][: action_space[0]])

        ac2_probs = self.sm_ac2(logits[0][action_space[0]: action_space[0] + action_space[1]])
        ac3_probs = self.sm_ac3(
            logits[0][action_space[0] + action_space[1]: action_space[0] + action_space[1] + action_space[2]])
        ac4_probs = self.sm_ac4(logits[0][
                                action_space[0] + action_space[1] + action_space[2]: action_space[0] + action_space[1] +
                                                                                     action_space[2] + action_space[3]])
        ac5_probs = self.sm_ac5(logits[0][
                                action_space[0] + action_space[1] + action_space[2] + action_space[3]: action_space[0] +
                                                                                                       action_space[1] +
                                                                                                       action_space[2] +
                                                                                                       action_space[3] +
                                                                                                       action_space[4]])

        probabilities = torch.cat((ac1_probs, ac2_probs, ac3_probs, ac4_probs, ac5_probs))

        actions, log_prob = self.md.log_prob_from_params(logits)

        return probabilities, actions, log_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        torch.load_state_dict(torch.load(self.checkpoint_file))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TetrisNetwork(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super(TetrisNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, state):
        temp = F.relu(self.fc1(state))
        action_values = F.tanh(self.fc2(temp))
        return action_values
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TetrisNetwork(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super(TetrisNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, state):
        temp = state.view(state.size(0), -1)
        temp = F.relu(self.fc1(temp))
        action_values = torch.tanh(self.fc2(temp))
        return action_values

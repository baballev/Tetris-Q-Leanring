import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TetrisNetwork(nn.Module):

    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
        super(TetrisNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_dim)

    def forward(self, state):
        temp = state.view(state.size(0), -1)
        temp = F.relu(self.fc1(temp))
        temp = F.relu(self.fc2(temp))
        action_values = self.fc3(temp)
        return action_values

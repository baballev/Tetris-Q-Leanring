import model
import utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TetrisTrainer():

    def __init__(self, capacity=500000, batch_size=128, gamma=0.999, lr=0.0001, epsilon_start=0.99, schedule_size=1000000):
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = lr
        self.epsilon = epsilon_start
        self.schedule_size = schedule_size

        self.q_network = model.TetrisNetwork(input_dim=200, hidden_size=32, output_dim=5).to(device) # The one we choose the actions from
        self.target_q_network = model.TetrisNetwork(input_dim=200, hidden_size=32, output_dim=5).to(device)  # The one we will be optimizing from (in part)
        self.update_target()

        self.memory = utils.ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, eps=1e-5)

    def optimize(self):
        state, new_state, action, reward = self.memory.sample(batch_size=self.batch_size)
        y = self.q_network(state)
        state_action_values = torch.stack([y[i, action[i]] for i in range(self.batch_size)])
        next_state_values = self.target_q_network(new_state).max(1)[0]
        expected_state_action_values = reward + self.gamma * next_state_values
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state):
        rand = random.random()  # real number between 0 and 1 uniformly distributed
        if rand > self.epsilon:
            action = torch.argmax(self.q_network(state))
        else:
            action = random.randint(0, 4)
        self.epsilon -= 1 / self.schedule_size  # ToDo: fix ugly code
        self.epsilon = max(0.02, self.epsilon)
        return action

    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


    # ToDo later:
    # - Save/checkpoint our training progess  =  saving the weights of the network to a file
    # - Load the weights
    # - Select best action
    # - Select actions that allow us to explore the state space
    # - Stop function at the end of training

import model
import utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import datetime
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TetrisTrainer():

    def __init__(self, capacity=100000, batch_size=128, gamma=0.999, lr=0.0001, epsilon_start=0.9, schedule_size=1000000, min_steps_training=100000):
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = lr
        self.epsilon = epsilon_start
        self.schedule_size = schedule_size
        self.min_steps_training = min_steps_training

        self.q_network = model.TetrisNetwork(input_dim=200, hidden_size1=64, hidden_size2=16, output_dim=5).to(device)  # The one we choose the actions from
        self.target_q_network = model.TetrisNetwork(input_dim=200, hidden_size1=64, hidden_size2=16, output_dim=5).to(device)  # The one we will be optimizing from (in part)
        self.update_target()

        self.memory = utils.PrioritizedMemory(capacity)
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.learning_rate)

        self.curr_reward = 0.0
        self.episode_rewards, self.episode_length, self.episode_loss, self.episode_q_values = [], [], [], []
        self.q_avg = torch.tensor(0.0, dtype=torch.float).to(device)
        self.count = 0
        self.curr_loss = 0.0

    def optimize(self):
        idx_batch, state, action, reward, new_state, _, weight = self.memory.sample(batch_size=self.batch_size)
        y = self.q_network(state)
        state_action_values = torch.stack([y[i, action[i]] for i in range(self.batch_size)])
        next_state_values = self.target_q_network(new_state).max(1)[0]
        expected_state_action_values = reward + self.gamma * next_state_values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        self.curr_loss += loss.mean().data
        self.optimizer.zero_grad()
        (weight * loss).mean().backward()
        self.optimizer.step()
        self.memory.update_priorities(idx_batch, loss.detach().cpu().numpy())

    def select_action(self, state, evaluation, step):
        with torch.no_grad():
            if evaluation:
                y = self.q_network(state)

                action = torch.argmax(y)
            else:
                rand = random.random()  # real number between 0 and 1 uniformly distributed
                if rand > self.epsilon:
                    y = self.q_network(state)
                    action = torch.argmax(y)
                    self.q_avg += y.mean()
                    self.count += 1
                else:
                    action = random.randint(0, 4)
                if step > self.min_steps_training:
                    self.epsilon -= 1 / self.schedule_size
                    self.epsilon = max(0.05, self.epsilon)
        return action

    def update_target(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def log(self, reward, done, step, episode):
        self.curr_reward += reward
        if done:
            self.episode_rewards.append(self.curr_reward)
            self.episode_length.append(step)
            if episode % 50 == 0 and episode > 0:
                print("End of episode " + str(episode) + " - " + str(torch.tensor(self.episode_length[-50:]).sum().item()/50) + " avg steps - Avg Episode Reward: " + str(torch.tensor(self.episode_rewards[-50:]).mean().item()))
                print("Avg Q-Table: " + str(self.q_avg.data/self.count) + " - Avg loss: " + str(self.curr_loss/50))
                self.episode_q_values.append(self.q_avg.data/self.count)
                self.episode_loss.append(self.curr_loss/50)

                self.q_avg = torch.tensor(0.0, dtype=torch.float).to(device)
                self.count = 0
                self.curr_loss = 0.0
            self.curr_reward = 0.0

    def save(self):
        t = str(datetime.datetime.now()).replace(":", "-") + ".pckl"
        with open("E:/Programmation/Python/tetrist_rl/checkpoints/" + t, 'wb') as f:
            pickle.dump(self, f)
            print("Saved trainer to file successfully to " + t)


def load(path):
    with open(path, 'rb') as f:
        tmp = pickle.load(f)
        tmp.curr_reward = 0.0
        return tmp
    # - Stop function at the end of training

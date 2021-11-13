import gym
import d3dshot
from PIL import Image
import torchvision
import torch
import torch.nn.functional as F
import utils

GBA_REGION = (1, 51, 241, 211)
GRID_REGION = (81, 51, 161, 211)


# ToDo: Put the tensors on the GPU and don't forget to go with no_grads when necessary

class TetrisGBAEnvironment(gym.Env):
    def __init__(self):
        super(TetrisGBAEnvironment, self).__init__()
        self.screen = d3dshot.create(capture_output="pytorch_float_gpu")  # ToDo: Try other devices for the preprocessing
        self.screen.capture(target_fps=60, region=GBA_REGION)

    def reset(self):  # ToDo: unpause the game, load F1 save, press enter and pause the game again.
        return torch.zeros((1, 20, 10), dtype=torch.float)

    def step(self, action, state):
        done = False
        self.perform_action(action)
        # wait a bit
        new_state = self.get_observation()
        reward = self.get_reward(state, new_state, action)
        return new_state, reward, done

    def get_observation(self):
        grid_pixels = self.screen.screenshot(region=GRID_REGION).permute(2, 0, 1)
        grid_pixels = torch.tensor(grid_pixels.mean(keepdim=True, dim=0) > 0.3, dtype=torch.float)
        grid = F.avg_pool2d(grid_pixels, (8, 8)) > 0.6  #  Should output a 20x10 grid
        return torch.tensor(grid, dtype=torch.float)

    def get_reward(self, state, new_state, action):  # ToDo: Think about a way to counter forever pressing rotate key
        height, new_height = self.compute_height(state), self.compute_height(new_state)
        if action == 4:
            if new_height < height:
                return (height - new_height)*torch.tensor(0.2, dtype=torch.float)
            if new_height > height:
                return new_height * torch.tensor(-0.05, dtype=torch.float)
            else:
                return torch.tensor(0.0, dtype=torch.float)
        else:
            return torch.tensor(-0.001, dtype=torch.float)

    @staticmethod
    def compute_height(state):
        height = state.sum(dim=2)
        height = torch.argmax(torch.tensor(torch.flip(height, dims=[0,1]) == 0.0, dtype=torch.int), dim=1).item() + 1
        return height

    def perform_action(self, action):
        if action == 0:  # DO NOTHING
            utils.pressAndHold('ctrl')
            for _ in range(12):
                utils.press('n')
            utils.release('ctrl')

        elif action == 1:  # ROTATE BLOCK
            utils.pressAndHold('x', 'ctrl')
            for i in range(12):
                if i == 1:
                    utils.release('x')
                utils.press('n')
            utils.release('ctrl')

        elif action == 2:  # LEFT
            utils.pressAndHold('h', 'ctrl')
            for i in range(12):
                if i == 1:
                    utils.release('h')
                utils.press('n')
            utils.release('ctrl')

        elif action == 3:  # RIGHT
            utils.pressAndHold('j', 'ctrl')
            for i in range(12):
                if i == 1:
                    utils.release('j')
                utils.press('n')
            utils.release('ctrl')

        else:  # DOWN
            utils.pressAndHold('g', 'ctrl')
            for i in range(12):
                if i == 1:
                    utils.release('g')
                utils.press('n')
            utils.release('ctrl')






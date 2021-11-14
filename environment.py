import gym
import d3dshot
from PIL import Image
import torchvision
import torch
import torch.nn.functional as F
import utils
import time

GBA_REGION = (1, 51, 241, 211)
GRID_REGION = (81, 51, 161, 211)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ToDo: Put the tensors on the GPU and don't forget to go with no_grads when necessary

class TetrisGBAEnvironment(gym.Env):
    def __init__(self):
        super(TetrisGBAEnvironment, self).__init__()
        self.screen = d3dshot.create(capture_output="pytorch_float_gpu")  # ToDo: Try other devices for the preprocessing
        self.screen.capture(target_fps=60, region=GRID_REGION)

    def reset(self):  # ToDo: unpause the game, load F1 save, press enter and pause the game again.
        utils.relaunch_routine()
        return torch.zeros((1, 20, 10), dtype=torch.float).to(device)

    def step(self, action, state):
        done = False
        self.perform_action(action)
        # wait a bit
        new_state = self.get_observation()
        reward = self.get_reward(state, new_state)
        return new_state, reward, done

    def get_observation(self):
        grid_pixels = self.screen.get_latest_frame().permute(2, 0, 1)
        grid_pixels = torch.ceil_(F.threshold(grid_pixels.mean(keepdim=True, dim=0), 0.3, 0.0))
        grid = torch.ceil_(F.threshold(F.avg_pool2d(grid_pixels, (8, 8)), 0.55, 0.0))
        return grid

    def get_reward(self, state, new_state):
        blocks, new_blocks = state.sum(), new_state.sum()
        print(blocks)
        print(new_blocks)
        if new_blocks - blocks < -8:
            return (blocks-new_blocks)/50
        elif blocks == new_blocks:
            return torch.tensor(-0.001, dtype=torch.float)
        elif new_blocks > blocks:
            return -(new_blocks - blocks)/100
        elif -8 < new_blocks - blocks < 0:  # Handling noise and compensate for the reward loss
            return -(new_blocks-blocks)/100 - torch.tensor(0.001, dtype=torch.float)

    '''
    @staticmethod
    def compute_height(state):
        height = state.sum(dim=2)
        height = torch.argmax(torch.clone(torch.flip(height, dims=[0,1]) == 0.0).to(torch.int), dim=1).item() + 1
        return height
    '''

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

    def stop(self):
        utils.release('ctrl')
        self.screen.stop()
        # ToDo: Close the VBA environment + save states





import gym
import d3dshot
from PIL import Image
import torchvision
import torch
import torch.nn.functional as F

GBA_REGION = (1, 51, 241, 211)
GRID_REGION = (81, 51, 161, 211)

class TetrisGBAEnvironment(gym.Env):
    def __init__(self):
        super(TetrisGBAEnvironment, self).__init__()
        self.screen = d3dshot.create(capture_output="pytorch_float_gpu")  # ToDo: Try other devices for the preprocessing
        self.screen.capture(target_fps=60, region=GBA_REGION)

    def step(self, action):

        #return new_state, reward
        pass

    def get_observation(self):
        grid_pixels = self.screen.screenshot(region=GRID_REGION).permute(2, 0, 1)
        #grid_pixels = torch.tensor((grid_pixels.mean(keepdim=True, dim=0) > 0.3), dtype=torch.float)
        #torchvision.transforms.ToPILImage()(grid_pixels).save("test_catpure2.png")
        grid_pixels = torch.tensor(grid_pixels.mean(keepdim=True, dim=0) > 0.3, dtype=torch.float)
        grid = F.avg_pool2d(grid_pixels, (8, 8)) > 0.6  #  Should output a 20x10 grid
        print(torch.tensor(grid, dtype=torch.int))


    def get_reward(self, state, new_state):
        pass


    def reset(self): # Will return state

        pass


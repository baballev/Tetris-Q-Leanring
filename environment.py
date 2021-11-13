import gym
import d3dshot
from PIL import Image

GBA_REGION = (1, 241, 61, 221)
GRID_REGION = (80, 160, 61, 221)

class TetrisGBAEnvironment(gym.Env):
    def __init__(self):
        super(TetrisGBAEnvironment, self).__init__()
        self.screen = d3dshot.create(capture_output="pil")  # ToDo: Try other devices for the preprocessing
        self.screen.capture(target_fps=60, region=GBA_REGION)

    def step(self, action):


        #return new_state, reward
        pass

    def get_observation(self):
        grid_pixels = self.screen.screenshot(region=GRID_REGION).permute(2, 0, 1)
        # DEBUG
        grid_pixels.save("test_image1.png")


    def get_reward(self, state, new_state):
        pass


    def reset(self): # Will return state

        pass


import gym
import d3dshot
from PIL import Image
import torchvision
import torch
import torch.nn.functional as F
import utils
import time
GBA_REGION = (1, 51, 241, 211)
GRID_REGION = (81, 51, 162, 211)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TetrisGBAEnvironment(gym.Env):
    def __init__(self, simulation=False):
        super(TetrisGBAEnvironment, self).__init__()
        self.simulation = simulation
        self.screen = d3dshot.create(capture_output="pytorch_float_gpu")  # ToDo: Try other devices for the preprocessing
        self.screen.capture(target_fps=60, region=GRID_REGION)
        if not self.simulation:
            self.process = utils.launch_environment_routine()

    def reset(self):  # ToDo: unpause the game, load F1 save, press enter and pause the game again.
        utils.relaunch_routine()
        return self.get_observation(0)[0]

    def step(self, action, state, step):
        self.perform_action(action)
        # wait a bit
        new_state, done = self.get_observation(step)
        if not done:
            reward = self.get_reward(state, new_state)
        if done:
            reward = torch.tensor(-1.0, dtype=torch.float).to(device)
        return new_state, torch.clip_(reward, -1.0, 1.0), done

    def get_observation(self, step):
        if not self.simulation:
            grid_pixels = self.screen.get_latest_frame().permute(2, 0, 1)
            if step > 10:
                done = grid_pixels[0, 50, -1] < 0.5
            else:
                done = False
            grid_pixels = torch.ceil_(F.threshold(grid_pixels.mean(keepdim=True, dim=0), 0.3, 0.0))
            grid = torch.ceil_(F.threshold(F.avg_pool2d(grid_pixels, (8, 8)), 0.6, 0.0))
            return grid, done
        else:
            pass

    @staticmethod
    def get_reward(state, new_state):
        blocks, new_blocks = state.sum(), new_state.sum()
        if new_blocks - blocks < -8:
            return (blocks-new_blocks)/50
        elif blocks == new_blocks:
            return torch.tensor(-0.001, dtype=torch.float).to(device)
        elif new_blocks > blocks:
            return -(new_blocks - blocks)/100
        elif -8 < new_blocks - blocks < 0:  # Handling noise and compensate for the reward loss
            return -(new_blocks-blocks)/100 - torch.tensor(0.001, dtype=torch.float)
        else:
            return torch.tensor(0.0, dtype=torch.float).to(device)



    def perform_action(self, action):
        if not self.simulation:
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
        else:
            pass


    def stop(self):
        utils.release('ctrl')
        self.screen.stop()
        try:
            self.process.kill()
            print('Killed process with pid' + str(self.process.pid))
        except Exception as e:
            print(e)

        # ToDo: Close the VBA environment + save states





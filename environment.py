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
        else:  # Blocks fall 2/1 block en alternance
            self.game_grid = torch.zeros((1, 21, 10)).to(device)
            self.alternance = True
            self.done = False

    def reset(self):  # ToDo: unpause the game, load F1 save, press enter and pause the game again.
        if self.simulation:
            self.game_grid = torch.zeros((1, 21, 10)).to(device)
        else:
            utils.relaunch_routine()
        return self.get_observation(0)[0]

    def step(self, action, state, step):
        self.perform_action(action)
        # wait for a bit
        new_state, done = self.get_observation(step)
        if not done:
            reward = self.get_reward(state, new_state)
        else:
            reward = torch.tensor(-1.0, dtype=torch.float).to(device)
        return new_state, torch.clip_(reward, -1.0, 1.0), done

    def get_observation(self, step):
        if not self.simulation:
            grid_pixels = self.screen.get_latest_frame().permute(2, 0, 1)
            done = grid_pixels[0, 50, -1] < 0.56
            grid_pixels = torch.ceil_(F.threshold(grid_pixels.mean(keepdim=True, dim=0), 0.3, 0.0))
            grid = torch.ceil_(F.threshold(F.avg_pool2d(grid_pixels, (8, 8)), 0.6, 0.0))
            return grid, done
        else:
            return self.game_grid, self.done
    @staticmethod
    def get_reward(state, new_state):
        blocks, new_blocks = state.sum(), new_state.sum()
        if new_blocks - blocks < -8:
            return (blocks-new_blocks)/40
        else:
            return torch.tensor(0.0, dtype=torch.float).to(device)

    def perform_action(self, action):
        if not self.simulation:
            if action == 0:  # DO NOTHING
                for _ in range(12):
                    utils.press('n')

            elif action == 1:  # ROTATE BLOCK
                utils.pressAndHold('x')
                for i in range(12):
                    if i >= 2:
                        utils.release('x')
                    utils.press('n')

            elif action == 2:  # LEFT
                utils.pressAndHold('h')
                for i in range(12):
                    if i >= 2:
                        utils.release('h')
                    utils.press('n')

            elif action == 3:  # RIGHT
                utils.pressAndHold('j')
                for i in range(12):
                    if i >= 2:
                        utils.release('j')
                    utils.press('n')

            else:  # DOWN
                utils.pressAndHold('g')
                for i in range(12):
                    if i >= 2:
                        utils.release('g')
                    utils.press('n')
        else:
            if action == 0:
                if self.alternance: # One block down
                    pass
                else: # Two blocks down
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





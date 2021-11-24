import gym
import d3dshot
from PIL import Image
import torchvision
import torch
import torch.nn.functional as F
import utils
import time
import simulation

GBA_REGION = (1, 51, 241, 211)
GRID_REGION = (81, 51, 162, 211)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TetrisGBAEnvironment(gym.Env):
    def __init__(self, simul=False, render=False):
        super(TetrisGBAEnvironment, self).__init__()
        if not simul:
            self.screen = d3dshot.create(
                capture_output="pytorch_float_gpu")  # ToDo: Try other devices for the preprocessing
            self.screen.capture(target_fps=60, region=GRID_REGION)
            self.process = utils.launch_environment_routine()
            self.simulation = None
        else:
            self.simulation = simulation.Simulation(render=render)

    def reset(self):  # ToDo: unpause the game, load F1 save, press enter and pause the game again.
        if self.simulation:
            return self.simulation.reset()
        else:
            utils.relaunch_routine()
            return self.get_observation(0)[0]

    def step(self, action, state, step):
        if simulation:
            new_state, done, reward = self.simulation.run_step(action)
        else:
            self.perform_action(action)
            new_state, done = self.get_observation(step)
            if not done:
                reward = self.get_reward(state, new_state)
            else:
                reward = torch.tensor(-1.0, dtype=torch.float).to(device)

        return new_state, reward, done

    def get_observation(self, step):
        grid_pixels = self.screen.get_latest_frame().permute(2, 0, 1)
        done = grid_pixels[0, 50, -1] < 0.56
        grid_pixels = torch.ceil_(F.threshold(grid_pixels.mean(keepdim=True, dim=0), 0.3, 0.0))
        grid = torch.ceil_(F.threshold(F.avg_pool2d(grid_pixels, (8, 8)), 0.6, 0.0))
        return grid, done

    @staticmethod
    def get_reward(state, new_state):
        blocks, new_blocks = state.sum(), new_state.sum()
        if new_blocks - blocks < -8:
            return (blocks-new_blocks)/40  # Todo: change to square the divided by 10 difference and then divide by 16 = 4²
        else:
            return torch.tensor(0.0, dtype=torch.float).to(device)

    def perform_action(self, action):
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

    def stop(self):
        utils.release('ctrl')
        self.screen.stop()
        try:
            self.process.kill()
            print('Killed process with pid' + str(self.process.pid))
        except Exception as e:
            print(e)

        # ToDo: Close the VBA environment + save states





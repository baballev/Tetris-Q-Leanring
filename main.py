from threading import Thread

import torch
import environment
import trainer
import time

from matplotlib import pyplot as plt
import numpy as np

import utils

MAX_STEPS = 500000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tetris_train(episode_nb=50000, min_steps_training=200, batch_size=250, gamma=0.9,
                lr=0.001, eps=0.99, schedule=500000, load=None, load_special_memory=None, simulation=False,
                 memory_capacity=20000, previous_t=0, evaluation=False):  # min_steps_training is considered to be smaller than the memory capacity
    env = environment.TetrisGBAEnvironment(simul=simulation, render=evaluation)
    if load is not None:
        tt = trainer.load(load)
        total_steps = previous_t
    else:
        tt = trainer.TetrisTrainer(batch_size=batch_size, gamma=gamma, lr=lr, epsilon_start=eps, schedule_size=schedule,
                               capacity=memory_capacity, min_steps_training=min_steps_training)
        total_steps = 0

    if load_special_memory is not None:
        tmp_tt = trainer.load(load_special_memory)
        tt.memory = tmp_tt.memory
        del tmp_tt
        total_steps = tt.memory.curr_size

    time.sleep(1)

    for episode in range(episode_nb):
        state = env.reset()
        for step in range(MAX_STEPS):
            action = tt.select_action(state, evaluation, total_steps)
            new_state, reward, done = env.step(action, state, step)
            tt.log(reward, done, step, episode)
            if not evaluation:
                tt.memory.push(new_state, action, reward, done)

            state = new_state
            total_steps += 1
            '''
            if total_steps % target_update_frequency == 0:
                tt.update_target()
                tt.save()
            '''
            if total_steps % 100000 == 0:
                print("Total number of steps: " + str(total_steps))
            if done:
                break

            # ToDo: Regular evaluation
        if episode > 5 and not evaluation:
            tt.optimize()
        if episode % 250 == 0 and not evaluation:
            tt.save()
            tt.update_target()



if __name__ == "__main__":
    tetris_train(simulation=True)

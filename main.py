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


def tetris_train(episode_nb=5000, min_steps_training=50000, target_update_frequency=30000, batch_size=32, gamma=0.999,
                lr=0.0001, eps=0.99, schedule=750000, load=None, load_special_memory=None, simulation=False,
                 memory_capacity=500000, previous_t=0, evaluation=False):  # min_steps_training is considered to be smaller than the memory capacity
    env = environment.TetrisGBAEnvironment(simul=simulation)
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

    th1, th2 = None, None
    time.sleep(1)

    for episode in range(episode_nb):
        state = env.reset()
        for step in range(MAX_STEPS):
            action = tt.select_action(state, evaluation, total_steps)
            new_state, reward, done = env.step(action, state, step)
            tt.log(reward, done, step, episode)
            if not evaluation:
                if th1 is not None:
                    th1.join()
                th1 = Thread(target=tt.memory.push, args=(state, new_state, action, reward))
                th1.start()
                if tt.memory.curr_size > min_steps_training:
                    if th2 is not None:
                        th2.join()
                    th2 = Thread(target=tt.optimize)
                    th2.start()

            state = new_state
            total_steps += 1
            if total_steps % target_update_frequency == 0:
                tt.update_target()
                tt.save()

            if done:
                break

            # ToDo: Regular evaluation
        if episode % 10 == 0:
            print("Total number of steps: " + str(total_steps))


if __name__ == "__main__":
    tetris_train(simulation=True, min_steps_training=100000, previous_t=4200000, episode_nb=500000, target_update_frequency=50000, schedule=2e6
                    ,load="E:/Programmation/Python/tetrist_rl/checkpoints/2021-11-19 18-22-34.717263.pckl")

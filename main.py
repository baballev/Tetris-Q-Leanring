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


def tetris_train(episode_nb=5000, min_steps_training=50000, target_update_frequency=30000, batch_size=128, gamma=0.999,
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

    th = None
    time.sleep(1)

    for episode in range(episode_nb):
        state = env.reset()
        for step in range(MAX_STEPS):
            #action = tt.select_action(state, evaluation, total_steps)
            print(state)
            action = input()
            new_state, reward, done = env.step(action, state, step)
            tt.log(reward, done, step, episode)
            if not evaluation:
                tt.memory.push(state, new_state, action, reward)
                if tt.memory.curr_size > min_steps_training:
                    if th is not None:
                        th.join()
                    th = Thread(target=tt.optimize)
                    th.start()

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
    tetris_train(simulation=True, min_steps_training=100000, episode_nb=500000, target_update_frequency=50000, schedule=1e6)

    #tt = trainer.load("E:/Programmation/Python/tetrist_rl/checkpoints/2021-11-18 10-22-06.499653.pckl")
    #print(tt.epsilon)
    '''
    tmp = torch.zeros((1, 20, 10), dtype=torch.float)
    #tmp[0, -2, : ] = torch.ones(10)
    #tmp[0, -1, : ] = torch.ones(10)
    #tmp[0, :, 5] = torch.ones(20)
    tmp[0, :, 6] = torch.ones(20)
    #tmp[0, 0, 6] = 0.0
    tmp[0, 1, 6] = 0.0
    tmp[0, 2, 6] = 0.0
    tmp[0, 0, 5] = 1.0
    tmp[0, 0, 4] = 1.0
    tmp[0, 0, 3] = 1.0

    #tmp[0, -2, 0] = 0.0
    #tmp[0, -1, 0] = 0.0

    #tmp[0, -2, 1] = 0.0
    #tmp[0, -1, 1] = 0.0

    print(tmp)
    tmp = tmp.to(device)
    print(tt.q_network(tmp))
    '''
    '''
    stat = tt.episode_rewards
    fig = plt.figure()
    ax = plt.subplot(111)
    n = len(stat)
    x = np.arange(0, n, 1)
    y = np.array(stat)
    ax.plot(x, y)
    y2 = np.convolve(y, np.ones(10), 'valid')/10
    ax.plot(x[:-9], y2)
    plt.show()
    '''

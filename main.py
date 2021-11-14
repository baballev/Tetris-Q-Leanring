import torch
import environment
import trainer
import time
import random

MAX_STEPS = 500000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tetris_train(episode_nb=5000, min_steps_training=50000, target_update_frequency=30000, batch_size=128, gamma=0.999,
                 lr=0.0001, eps=0.99, schedule=100000,
                 memory_capacity=500000):  # min_steps_training is considered to be smaller than the memory capacity
    env = environment.TetrisGBAEnvironment()
    tt = trainer.TetrisTrainer(batch_size=batch_size, gamma=gamma, lr=lr, epsilon_start=eps, schedule_size=schedule,
                               capacity=memory_capacity)
    total_steps = 0

    for episode in range(episode_nb):
        state = env.reset()

        for step in range(MAX_STEPS):
            action = tt.select_action(state)
            new_state, reward, done = env.step(action, state)
            print(reward)
            tt.memory.push(state, new_state, action, reward)
            if tt.memory.curr_size > min_steps_training:
                tt.optimize()

            if done:
                break

            state = new_state
            total_steps += 1
            if total_steps % target_update_frequency == 0:
                tt.update_target()

            # ToDo: Load and save weights,  pickle trainer
            # ToDo: Logging
            # ToDo: Regular evaluation
            # ToDo: tdqm episode number

if __name__ == "__main__":
    tetris_train()

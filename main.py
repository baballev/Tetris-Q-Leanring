import torch
import environment
import trainer
import time
import random

MAX_STEPS = 500000


def tetris_train(episode_nb=5000, min_steps_training=50000, target_update_frequency=30000):  # min_steps_training is considered to be smaller than the memory capacity
    env = environment.TetrisGBAEnvironment()
    tt = trainer.TetrisTrainer()
    total_steps = 0

    for episode in range(episode_nb):
        state = env.reset()

        for step in range(MAX_STEPS):
            #action = function(state)   # ToDo
            action = random.randint(0, 4)
            new_state, reward, done = env.step(action, state)

            tt.memory.push(state, new_state, action, reward)
            if tt.memory.curr_size > min_steps_training:
                trainer.optimize()

            if done:
                break

            state = new_state
            total_steps += 1
            if total_steps % target_update_frequency == 0:
                tt.update_target()

            # ToDo: Load and save weights,  pickle trainer

if __name__ == "__main__":
    tetris_train()

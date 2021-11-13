import torch
import environment
import trainer
import time
import random

MAX_STEPS = 500000


def tetris_train(episode_nb=5000):
    env = environment.TetrisGBAEnvironment()
    tt = trainer.TetrisTrainer()

    for episode in range(episode_nb):
        state = env.reset()

        for step in range(MAX_STEPS):
            #action = function(state)   # ToDo
            action = random.randint(0, 4)

            new_state, reward, done = env.step(action, state)
            print(reward)
            # ToDo: Learn
            #trainer.learn(state, new_state, action, reward)

            if done:
                break

            state = new_state


if __name__ == "__main__":
    tetris_train()

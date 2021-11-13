import torch
import environment
import trainer
import time

MAX_STEPS = 500000



def tetris_train(episode_nb=5000):
    env = environment.TetrisGBAEnvironment()
    tt = trainer.TetrisTrainer()

    for episode in range(episode_nb):
        state = env.reset()


        for step in range(MAX_STEPS):
            action = function(state)   # ToDo

            new_state, reward, done = env.step(action)


            # ToDo: Learn
            trainer.learn(state, new_state, action, reward)

            if done:
                break

            state = new_state



if __name__ == "__main__":
    #tetris_train()
    env = environment.TetrisGBAEnvironment()

    time.sleep(10)
    env.get_observation()
    

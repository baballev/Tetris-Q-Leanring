import torch
import environment
import trainer
import time
import random
import tqdm

MAX_STEPS = 500000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tetris_train(episode_nb=5000, min_steps_training=50000, target_update_frequency=30000, batch_size=128, gamma=0.999,
                 lr=0.0001, eps=0.99, schedule=100000, load=None,
                 memory_capacity=1000000):  # min_steps_training is considered to be smaller than the memory capacity
    env = environment.TetrisGBAEnvironment()
    if load is not None:
        tt = trainer.load(load)
    else:
        tt = trainer.TetrisTrainer(batch_size=batch_size, gamma=gamma, lr=lr, epsilon_start=eps, schedule_size=schedule,
                               capacity=memory_capacity)
    total_steps = 0
    time.sleep(1)

    for episode in tqdm.tqdm(range(episode_nb)):
        state = env.reset()

        for step in range(MAX_STEPS):
            action = tt.select_action(state)
            new_state, reward, done = env.step(action, state)
            tt.log(reward, done, step, episode)
            tt.memory.push(state, new_state, action, reward)
            if tt.memory.curr_size > min_steps_training:
                tt.optimize()

            state = new_state
            total_steps += 1
            if total_steps % target_update_frequency == 0:
                tt.update_target()
                tt.save()

            if done:
                break

            # ToDo: Regular evaluation


if __name__ == "__main__":
    tetris_train(load="E:/Programmation/Python/tetrist_rl/checkpoints/2021-11-14 15-00-15.199165.pckl")nnnnnnnn

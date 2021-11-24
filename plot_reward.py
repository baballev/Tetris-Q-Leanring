import trainer
import matplotlib.pyplot as plt
import numpy as np

tt = trainer.load("E:/Programmation/Python/tetrist_rl/checkpoints/2021-11-22 08-42-52.597609.pckl")
print(tt.epsilon)
stat = tt.episode_rewards
fig = plt.figure()
ax = plt.subplot(111)
n = len(stat)
x = np.arange(0, n, 1)
y = np.array(stat)
ax.plot(x, y)
y2 = np.convolve(y, np.ones(100), 'valid') / 100
ax.plot(x[:-99], y2)
plt.show()
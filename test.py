import gym
from uuv_env.uuv_env import UUVEnv
import numpy as np

env = UUVEnv()
obs = env.reset()

done = False
while not done:
    action = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
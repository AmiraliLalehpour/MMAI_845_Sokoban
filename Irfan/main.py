#pip install gym==0.21.00
#pip install gym-sokoban
#pip install pygame
#pip install pyglet

import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from sokoban_env import SokobanEnv

#Import files with different models
from dqn import train_dqn
from sarsa import sarsa
import sys



print(gym.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
#TensorFlow version: 2.13.0
#NumPy version: 1.24.3
print(sys.version)

# Create the Sokoban environment
seed = 123
#env_name = 'Sokoban-v2'
#env = gym.make(env)

env = SokobanEnv(dim_room=(10, 10), max_steps=120, num_boxes=3, num_gen_steps=30, reset=True)
env.seed(seed)


#How to visualize the environment
env.render(mode='human')
#env.render(mode='rgb_array')
env.render(mode='human')
# Action lookup
ACTION_LOOKUP = env.unwrapped.get_action_lookup()

# SARSA parameters
num_episodes = 100
learning_rate = 0.3
discount_factor = 0.99
exploration_prob = 0.05
# Define the minimum reward threshold to consider the agent has learned the task
min_reward_threshold = -50  # Adjust this threshold based on your task
# Define the maximum number of episodes for training
max_episodes = 1000

# Run SARSA algorithm
#sarsa(env, num_episodes, learning_rate, discount_factor, exploration_prob, min_reward_threshold)


# DQN parameters
num_episodes = 100
batch_size = 32
learning_rate = 1e-4
discount_factor = 0.99
exploration_fraction = 0.2
exploration_final_eps = 0.01

print("Hello Observation shape before preprocessing:", env.observation_space.shape)
# Run the DQN with CNN policy and get the average total reward
# Example usage
reward = train_dqn(env, num_episodes=10000, learning_rate=1e-3, discount_factor=0.99, exploration_fraction=0.1,exploration_final_eps=0.01, target_model_update=1e-2)

print("Average Total Reward:", avg_total_reward)

env.close()
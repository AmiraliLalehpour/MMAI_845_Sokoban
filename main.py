#pip install gym==0.21.00
#pip install gym-sokoban
#pip install pygame
#pip install pyglet

import gym
import numpy as np
import gym_sokoban
import time
import tensorflow as tf
# from sokoban_env import SokobanEnv
import sys

print(gym.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
#TensorFlow version: 2.13.0
#NumPy version: 1.24.3
print(sys.version)


#SARSA Implementation
# Create the Sokoban environment
seed = 123
env_name = 'Sokoban-v2'
env = gym.make(env_name)
env.seed(seed)

#How to visualize the environment
#env.render(mode='human')
#env.render(mode='rgb_array')
env.render(mode='tiny_rgb_array')
# Action lookup
ACTION_LOOKUP = env.unwrapped.get_action_lookup()
# Convert state to tuple representation (for tabular SARSA)
def state_to_tuple(state):
    return tuple(state.ravel())

"""SARSA IMPLEMENTATION"""
# SARSA parameters
num_episodes = 100
learning_rate = 0.3
discount_factor = 0.99
exploration_prob = 0.05

# Q-table initialization
q_table = {}
state_tuple = 0
#state = env.reset()

# Define the maximum number of episodes for training
max_episodes = 1000
# Define the minimum reward threshold to consider the agent has learned the task
min_reward_threshold = -150  # Adjust this threshold based on your task


# SARSA algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # Initialize Q-values for the current state if not present
    if state_tuple not in q_table:
        q_table[state_tuple] = np.zeros(env.action_space.n)

    # Choose the initial action based on epsilon-greedy policy
    if np.random.rand() < exploration_prob:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state_tuple])

    while not done:
        #env.render(mode='tiny_human')
        time.sleep(1)

        # Take the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        # Initialize Q-values for the next state if not present
        if next_state_tuple not in q_table:
            q_table[next_state_tuple] = np.zeros(env.action_space.n)

        # Choose the next action based on epsilon-greedy policy
        if np.random.rand() < exploration_prob:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state_tuple])

        # SARSA Q-value update
        q_value = q_table[state_tuple][action]
        next_q_value = q_table[next_state_tuple][next_action]
        q_table[state_tuple][action] = q_value + learning_rate * (reward + discount_factor * next_q_value - q_value)

        state = next_state.copy()  # Copy the next_state into the state variable
        state_tuple = next_state_tuple
        action = next_action
        total_reward += reward
        print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))

        if done:
            print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))
            env.render(mode='human')
            break
        if total_reward <= min_reward_threshold:
            print("Training completed! Satisfactory reward achieved.")
            env.render(mode='human')
            break
env.close()
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import gym_sokoban
import matplotlib.pyplot as plt
from sokoban_env import SokobanEnv
from custom_sokoban_env import my_sokoban_env
import time
import copy
import random 
import itertools


# # Check System Requirements

# In[2]:


# pip install gym==0.21.0


# In[3]:


import gym

print(gym.__version__)


# In[4]:


import sys

print(sys.version)


# # Define the environment and number of boxes

# In[5]:


# # Create the Sokoban environment from sokoban versions.
# env_name = 'Sokoban-v2'
# game_env = gym.make(env_name)


# In[6]:


# Create the Sokoban environment custom
game_env = my_sokoban_env(dim_room=(10, 10), num_boxes=3)


# In[7]:


# Function to convert state to a tuple
def state_to_tuple(state):
    return tuple(state.reshape(-1))

# Save the original state of the environment
# initial_state = game_env.second_reset()
# initial_state_tuple = state_to_tuple(initial_state)
# game_env.render(mode='human')


# In[8]:


# Action lookup
ACTION_LOOKUP = game_env.unwrapped.get_action_lookup()
# Convert state to tuple representation (for tabular SARSA)
def state_to_tuple(state):
    return tuple(state.ravel())


# # Save One Initial State for Consistancy

# ![three_box_env.JPG](attachment:three_box_env.JPG)

# In[9]:


# We need to save the exact one we see below so comment this when selected the desired env topology
# initial_agent_position = game_env.player_position
# initial_box_mapping = game_env.box_mapping
# initial_room_fixed = game_env.room_fixed
# initial_room_state = game_env.room_state


# In[10]:


initial_room_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 4, 1, 1, 1, 0],
                               [0, 0, 5, 4, 1, 1, 1, 2, 1, 0],
                               [0, 0, 0, 0, 1, 1, 2, 1, 1, 0],
                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 2, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# In[11]:


initial_room_fixed = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 0, 1, 1, 1, 1, 1, 2, 1, 0],
                               [0, 0, 0, 0, 1, 1, 2, 1, 1, 0],
                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 2, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# In[12]:


initial_box_mapping = np.array({(3, 7): (7, 5), (4, 6): (3, 3), (6, 5): (2, 5)})


# In[13]:


initial_agent_position = np.array([3,2])


# # Implement the RL Algorithms

# ## SARSA Algorithm

# In[14]:


num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.05

sarsa_q_table = {} 

# Training the agent
game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
    initial_box_mapping=initial_box_mapping,
    initial_room_fixed=initial_room_fixed,
    initial_room_state=initial_room_state)
# SARSA algorithm
for episode in range(num_episodes):

    #print('starting episode', episode+1)
    state = game_env.second_reset() 
    # Updated the initial variables with the current state returned by second_reset()
    state_tuple = state_to_tuple(state)
    done = False
    sarsa_total_reward = 0

    # Initialize Q-values for the current state if not present
    if state_tuple not in sarsa_q_table:
        sarsa_q_table[state_tuple] = np.zeros(game_env.action_space.n)

    # Choose the initial action based on epsilon-greedy policy
    if np.random.rand() < exploration_prob :
        action = game_env.action_space.sample()
    else:
        action = np.argmax(sarsa_q_table[state_tuple])

    while not done:
        game_env.render(mode='human')

        # Take the chosen action
        next_state, reward, done, _ = game_env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        # Initialize Q-values for the next state if not present
        if next_state_tuple not in sarsa_q_table:
            sarsa_q_table[next_state_tuple] = np.zeros(game_env.action_space.n)

        # Choose the next action based on epsilon-greedy policy
        if np.random.rand() < exploration_prob :
            next_action = game_env.action_space.sample()
        else:
            next_action = np.argmax(sarsa_q_table[next_state_tuple])

        # SARSA Q-value update
        q_value = sarsa_q_table[state_tuple][action]
        next_q_value = sarsa_q_table[next_state_tuple][next_action]
        sarsa_q_table[state_tuple][action] = q_value + learning_rate * (reward + discount_factor * next_q_value - q_value)

        state = next_state.copy()  # Copy the next_state into the state variable
        state_tuple = next_state_tuple
        action = next_action
        sarsa_total_reward += reward

        if done:
            print("Episode: {}, Total Reward: {}".format(episode + 1, sarsa_total_reward))
            break


# In[15]:


# Function to test the agent's performance in one final test episode using the learned Q-table
def test_agent(q_table, game_env):
    state = game_env.second_reset() 
    # Updated the initial variables with the current state returned by second_reset()
    state_tuple = state_to_tuple(state)
    done = False
    total_reward = 0

    while not done:
        # Choose the best action based on the Q-table
        action = np.argmax(q_table[state_tuple])
        game_env.render(mode='human')
        next_state, reward, done, _ = game_env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        state = next_state.copy()
        state_tuple = next_state_tuple
        total_reward += reward

    return total_reward

# Initialize the game environment
game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
                          initial_box_mapping=initial_box_mapping,
                          initial_room_fixed=initial_room_fixed,
                          initial_room_state=initial_room_state)

# Test the agent using the trained Q-table for one final episode
final_reward = test_agent(sarsa_q_table, game_env)

# Print the final reward obtained in the test episode
print("Final Test Episode Reward:", final_reward)


# ## Q-Learning Algorithm

# In[16]:


# Q-Learning
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.05

q_table = {} 

# Training the agent
game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
    initial_box_mapping=initial_box_mapping,
    initial_room_fixed=initial_room_fixed,
    initial_room_state=initial_room_state)
# Q-Learning algorithm
for episode in range(num_episodes):

    #print('starting episode', episode+1)   
    state = game_env.second_reset() 
    # Updated the initial variables with the current state returned by second_reset()
    state_tuple = state_to_tuple(state)
    done = False
    total_reward = 0

    # Initialize Q-values for the current state if not present
    if state_tuple not in q_table:
        q_table[state_tuple] = np.zeros(game_env.action_space.n)

    # Choose the initial action based on epsilon-greedy policy
    if np.random.rand() < exploration_prob :
        action = game_env.action_space.sample()
    else:
        action = np.argmax(q_table[state_tuple])

    while not done:
        game_env.render(mode='human')

        # Take the chosen action
        next_state, reward, done, _ = game_env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        # Initialize Q-values for the next state if not present
        if next_state_tuple not in q_table:
            q_table[next_state_tuple] = np.zeros(game_env.action_space.n)

        # Choose the next action based on epsilon-greedy policy
        if np.random.rand() < exploration_prob :
            next_action = game_env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state_tuple])

        # SARSA Q-value update
        q_value = q_table[state_tuple][action]
        max_next_q_value = np.max(q_table[next_state_tuple])
        q_table[state_tuple][action] = q_value + learning_rate * (reward + discount_factor * max_next_q_value - q_value)

        state = next_state.copy()  # Copy the next_state into the state variable
        state_tuple = next_state_tuple
        action = next_action
        total_reward += reward

        if done:
            print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))
            break


# In[17]:


# Function to test the agent's performance in one final test episode using the learned Q-table
def test_agent(q_table, game_env):
    state = game_env.second_reset() 
    # Updated the initial variables with the current state returned by second_reset()
    state_tuple = state_to_tuple(state)
    done = False
    total_reward = 0

    while not done:
        # Choose the best action based on the Q-table
        action = np.argmax(q_table[state_tuple])
        game_env.render(mode='human')
#         time.sleep(0.01)
        next_state, reward, done, _ = game_env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        state = next_state.copy()
        state_tuple = next_state_tuple
        total_reward += reward

    return total_reward

# Initialize the game environment
game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
                          initial_box_mapping=initial_box_mapping,
                          initial_room_fixed=initial_room_fixed,
                          initial_room_state=initial_room_state)

# Test the agent using the trained Q-table for one final episode
final_reward = test_agent(q_table, game_env)

# Print the final reward obtained in the test episode
print("Final Test Episode Reward for Q-Learning:", final_reward)


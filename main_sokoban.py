#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gym
import gym_sokoban
import matplotlib.pyplot as plt
from custom_sokoban_env import my_sokoban_env
import sokoban_tabular
import time
import sys
import random 
import itertools


# # Check System Requirements

# In[ ]:


# Install these if needed
# pip install gym==0.21.0
# !pip install pyglet
#!pip install imageio


# In[ ]:


print(gym.__version__)


# In[ ]:


print(sys.version)


# # Define the environment and number of boxes

# In[ ]:


# # Create the Sokoban environment from sokoban versions.
# env_name = 'Sokoban-v2'
# game_env = gym.make(env_name)


# In[ ]:


# Create the Sokoban environment custom
# game_env = my_sokoban_env(dim_room=(10, 10), num_boxes=3)


# In[ ]:


# game_env.reset()
# game_env.render(mode='human')


# In[ ]:


# Function to convert state to a tuple
def state_to_tuple(state):
    return tuple(state.reshape(-1))

# Save the original state of the environment, used first deciding on a env to keep constant
# initial_state = game_env.reset()
# initial_state_tuple = state_to_tuple(initial_state)
# game_env.render(mode='human')


# In[ ]:


# Action lookup
# ACTION_LOOKUP = env.unwrapped.get_action_lookup()
# Convert state to tuple representation (for tabular SARSA)
def state_to_tuple(state):
    return tuple(state.ravel())


# # Save One Initial State for Consistancy

# ![three_box_env.JPG](attachment:three_box_env.JPG)

# ![4_box_env.png](attachment:4_box_env.png)

# In[ ]:


# # We need to save the exact one we see below so comment this when selected the desired env topology
# initial_agent_position = game_env.player_position
# initial_box_mapping = game_env.box_mapping
# initial_room_fixed = game_env.room_fixed
# initial_room_state = game_env.room_state


# In[ ]:


# 3 Box environment
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

initial_box_mapping = np.array({(3, 7): (7, 5), (4, 6): (3, 3), (6, 5): (2, 5)})

initial_agent_position = np.array([3,2])


# In[ ]:


# # 4 Box environment
# initial_room_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 2, 1, 1, 1, 1, 0],
#        [0, 0, 0, 5, 4, 1, 0, 4, 1, 0],
#        [0, 0, 0, 0, 2, 1, 0, 1, 1, 0],
#        [0, 0, 0, 1, 4, 1, 1, 0, 0, 0],
#        [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# initial_room_fixed = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 2, 1, 1, 1, 1, 0],
#        [0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
#        [0, 0, 0, 0, 2, 1, 0, 1, 1, 0],
#        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
#        [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# initial_box_mapping = np.array({(1, 4): (2, 7), (3, 4): (4, 4), (5, 4): (7, 4), (5, 5): (2, 4)})

# initial_agent_position = np.array([2, 3])


# In[ ]:


game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
                        initial_box_mapping=initial_box_mapping,
                        initial_room_fixed=initial_room_fixed,
                        initial_room_state=initial_room_state)


# In[ ]:


## if you want to see the generated shape
# game_env.reset()
# game_env.render(mode='human')


# # Plots 7 Functions

# In[ ]:


def plot_rew(title, rew_list, label):
    plt.ioff()
    plt.title("Sokoban Warehouse: {}".format(title))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rew_list, label=label)
    plt.legend()
    plt.show()


# In[ ]:


# Function to test the agent's performance in one final test episode using the learned Q-table
def test_agent(q_table, game_env):
    state = game_env.reset() 
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
        game_env.render(mode='human')
        time.sleep(1)
    return total_reward


# # Run Algorithms

# In[ ]:


if __name__=='__main__':

    # Flags to run SARSA and/or Q-learning
    run_sarsa = True #True
    run_q_learning = True

    # Create the environment and policy
    env = my_sokoban_env(initial_agent_position=initial_agent_position,
                        initial_box_mapping=initial_box_mapping,
                        initial_room_fixed=initial_room_fixed,
                        initial_room_state=initial_room_state)
    if run_sarsa:
        print("------ Running SARSA ------")
        # Parameters for SARSA algorithm
        num_episodes = 300
        learning_rate = 0.5
        discount_factor = 0.99
        exploration_prob = 0.01
        
        sarsa_rews, sarsa_table = sokoban_tabular.sarsa(env, num_episodes=num_episodes, 
                                                 learning_rate=learning_rate, 
                                                 discount_factor=discount_factor, 
                                                 exploration_prob=exploration_prob)        
        plot_rew('SARSA', sarsa_rews, label = 'SARSA')
        # Test the agent using the trained Q-table for one final episode
        final_reward = test_agent(sarsa_table, env)
        print("The return of your SARSA solution is {}".format(final_reward))
        print("------ Finished running SARSA ------")
        
    if run_q_learning:
        print("------ Running Q-learning ------")
        num_episodes = 300
        learning_rate = 0.5
        discount_factor = 0.99
        exploration_prob = 0.01
        
        ql_rews, ql_table = sokoban_tabular.q_learning(env, num_episodes=num_episodes, 
                                                 learning_rate=learning_rate, 
                                                 discount_factor=discount_factor, 
                                                 exploration_prob=exploration_prob)

        
        plot_rew('Q-Learning', ql_rews, label = 'Q-Learning')
        final_reward = test_agent(ql_table, env)
        print("The return of your Q-learning solution is {}".format(final_reward))
        print("------ Finished running Q-learning ------")


# In[ ]:


plot_rew('SARSA', sarsa_rews, label = 'SARSA')


# In[ ]:


plot_rew('Q-Learning', ql_rews, label = 'Q-Learning')


# In[ ]:





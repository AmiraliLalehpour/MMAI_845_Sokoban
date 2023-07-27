#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import time


# In[10]:


def state_to_tuple(state):
    return tuple(state.reshape(-1))


# In[11]:


# Function to train the agent using SARSA algorithm
def sarsa(env, num_episodes=100, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.05):
    q_table = {}
    total_reward_list = []

    for episode in range(num_episodes):
        state = env.second_reset()
        state_tuple = state_to_tuple(state)
        done = False
        reward_list = 0
# greedy action:
        if state_tuple not in q_table:
            q_table[state_tuple] = np.zeros(env.action_space.n)

        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_tuple])
# start learning in episode:
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_state_tuple = state_to_tuple(next_state)

            if next_state_tuple not in q_table:
                q_table[next_state_tuple] = np.zeros(env.action_space.n)

            if np.random.rand() < exploration_prob:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state_tuple])

            q_value = q_table[state_tuple][action]
            next_q_value = q_table[next_state_tuple][next_action]
            q_table[state_tuple][action] = q_value + learning_rate * (reward + discount_factor * next_q_value - q_value)

            state = next_state.copy()
            state_tuple = next_state_tuple
            action = next_action
            reward_list += reward
#             env.render(mode='human')

            if done:
                print("Episode: {}, Total Reward: {}".format(episode + 1, reward_list))
                total_reward_list.append(reward_list)  # Append the reward for this episode to the list
                break

    # Print the entire reward list after training
    print("All Rewards:", total_reward_list)
    return total_reward_list, q_table



# In[12]:


# Function to train the agent using Q-Learning algorithm
def q_learning(env, num_episodes=100, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.05):
    q_table = {}
    total_reward_list = []

    for episode in range(num_episodes):
        state = env.second_reset()
        state_tuple = state_to_tuple(state)
        done = False
        reward_list = 0
# greedy action:
        if state_tuple not in q_table:
            q_table[state_tuple] = np.zeros(env.action_space.n)

        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_tuple])
# start learning in episode:
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_state_tuple = state_to_tuple(next_state)

            if next_state_tuple not in q_table:
                q_table[next_state_tuple] = np.zeros(env.action_space.n)

            if np.random.rand() < exploration_prob:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state_tuple])

            q_value = q_table[state_tuple][action]
            max_next_q_value = np.max(q_table[next_state_tuple])
            q_table[state_tuple][action] = q_value + learning_rate * (reward + discount_factor * max_next_q_value - q_value)

            state = next_state.copy()
            state_tuple = next_state_tuple
            action = next_action
            reward_list += reward
#             env.render(mode='human')

            if done:
                print("Episode: {}, Total Reward: {}".format(episode + 1, reward_list))
                total_reward_list.append(reward_list)
                break

    # Print the entire reward list after training
    print("All Rewards:", total_reward_list)
    
    return total_reward_list, q_table




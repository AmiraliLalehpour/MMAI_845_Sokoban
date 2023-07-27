"""SARSA IMPLEMENTATION"""
import numpy as np
import time


# Convert state to tuple representation (for tabular SARSA)
def state_to_tuple(state):
    return tuple(state.ravel())

# SARSA algorithm
def sarsa(env, num_episodes, learning_rate, discount_factor, exploration_prob, min_reward_threshold):
    # Q-table initialization
    q_table = {}
    state_tuple = 0
    state = env.reset()
    for episode in range(num_episodes):
        state = env.reset_new()
        done = False
        total_reward = 0
        step = 0

        # Initialize Q-values for the current state if not present
        if state_tuple not in q_table:
            q_table[state_tuple] = np.zeros(env.action_space.n)

        # Choose the initial action based on epsilon-greedy policy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_tuple])

        while not done:
            env.render(mode='human')
            time.sleep(0.01)

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
            step += 1
            print("Episode: {}, Step: {}, Total Reward: {}".format(episode + 1, step, total_reward))


            if done:
                print("Episode: {}, Step: {}, Total Reward: {}".format(episode + 1, step, total_reward))
                env.render(mode='human')
                break
            #elif total_reward <= min_reward_threshold:
             #   print("Training completed! Negative reward threshold exceded.")
              #  env.render(mode='human')
               # break
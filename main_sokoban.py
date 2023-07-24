#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import gym
import gym_sokoban
import time
import copy
import random 
import itertools


# In[30]:


# pip install gym==0.21.0


# In[ ]:


import gym

print(gym.__version__)


# In[32]:


from sokoban_env import SokobanEnv
import sys

print(sys.version)


# In[34]:


# Create the Sokoban environment
env_name = 'Sokoban-v2'
game_env = gym.make(env_name)


# In[ ]:


# Function to convert state to a tuple
def state_to_tuple(state):
    return tuple(state.reshape(-1))

# Save the original state of the environment
initial_state = game_env.reset()
initial_state_tuple = state_to_tuple(initial_state)
game_env.render(mode='human')


# In[ ]:


# Action lookup
ACTION_LOOKUP = game_env.unwrapped.get_action_lookup()
# Convert state to tuple representation (for tabular SARSA)
def state_to_tuple(state):
    return tuple(state.ravel())


# In[ ]:


initial_agent_position = game_env.player_position
initial_box_mapping = game_env.box_mapping
initial_room_fixed = game_env.room_fixed
initial_room_state = game_env.room_state


# In[ ]:


initial_room_state


# In[ ]:


initial_box_mapping


# In[ ]:


game_env.box_mapping


# In[ ]:


import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from room_utils import generate_room
from render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np

from gym.envs.registration import register

from pickletools import UP_TO_NEWLINE
from stat import UF_OPAQUE
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
# np.random.seed(4)

class my_sokoban_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True,
                 random_seed=7,
                 initial_agent_position=None,
                 initial_box_mapping=None,
                 initial_room_fixed=None,
                 initial_room_state=None,
                 second_player_added = False
                ):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0
        self.random_seed = random_seed
        self.second_player_added = False

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        # Add new attributes for storing initial states
        self.initial_agent_position = initial_agent_position
        self.initial_box_mapping = initial_box_mapping
        self.initial_room_fixed = initial_room_fixed
        self.initial_room_state = initial_room_state

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)
    
    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player,
                
            )

        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation
    
    def second_reset(self, second_player=False, render_mode='rgb_array'):
        try:

#             self.room_fixed, self.room_state, self.box_mapping = initial_room_fixed,initial_room_state, initial_box_mapping
            # Reset the environment to the initial states
            self.player_position = self.initial_agent_position.copy()
            self.box_mapping = self.initial_box_mapping.copy()
            self.room_fixed = self.initial_room_fixed.copy()
            self.room_state = self.initial_room_state.copy()

        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']


# In[ ]:


import rl_sarsa_sokoban
# from custom_sokoban_env import my_sokoban_env
import matplotlib.pyplot as plt
import numpy as np
import gym


# SARSA Algorithm

num_episodes= 500
alpha=0.3
gamma=1
epsilon=0.05

q_table = {} 

# Training the agent
game_env = my_sokoban_env(initial_agent_position=initial_agent_position,
    initial_box_mapping=initial_box_mapping,
    initial_room_fixed=initial_room_fixed,
    initial_room_state=initial_room_state)
# SARSA algorithm
for episode in range(num_episodes):

    print('starting episode', episode+1)
    
    
    state = game_env.second_reset() 
    # Update the initial variables with the current state returned by second_reset()
    initial_agent_position = game_env.initial_agent_position
    initial_box_mapping = game_env.initial_box_mapping
    initial_room_fixed = game_env.initial_room_fixed
    initial_room_state = game_env.initial_room_state
    print(initial_box_mapping)
    state_tuple = state_to_tuple(state)
    done = False
    total_reward = 0

    # Initialize Q-values for the current state if not present
    if state_tuple not in q_table:
        q_table[state_tuple] = np.zeros(game_env.action_space.n)

    # Choose the initial action based on epsilon-greedy policy
    if np.random.rand() < epsilon:
        action = game_env.action_space.sample()
    else:
        action = np.argmax(q_table[state_tuple])

    while not done:
#         game_env.render(mode='human')
        time.sleep(1)

        # Take the chosen action
        next_state, reward, done, _ = game_env.step(action)
        next_state_tuple = state_to_tuple(next_state)

        # Initialize Q-values for the next state if not present
        if next_state_tuple not in q_table:
            q_table[next_state_tuple] = np.zeros(game_env.action_space.n)

        # Choose the next action based on epsilon-greedy policy
        if np.random.rand() < epsilon:
            next_action = game_env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state_tuple])

        # SARSA Q-value update
        q_value = q_table[state_tuple][action]
        next_q_value = q_table[next_state_tuple][next_action]
        q_table[state_tuple][action] = q_value + alpha * (reward + gamma * next_q_value - q_value)

        state = next_state.copy()  # Copy the next_state into the state variable
        state_tuple = next_state_tuple
        action = next_action
        total_reward += reward

        if done:
            print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))
            break


# In[ ]:


import gym
import numpy as np


# Function to test the agent's performance in one final test episode using the Q-table
def test_agent(q_table, game_env):
    state = game_env.second_reset() 
    # Update the initial variables with the current state returned by second_reset()
    initial_agent_position = game_env.initial_agent_position
    initial_box_mapping = game_env.initial_box_mapping
    initial_room_fixed = game_env.initial_room_fixed
    initial_room_state = game_env.initial_room_state
    print(initial_box_mapping)
    state_tuple = state_to_tuple(state)
    done = False
    total_reward = 0

    while not done:
        # Choose the best action based on the Q-table
        action = np.argmax(q_table[state_tuple])

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
print("Final Test Episode Reward:", final_reward)


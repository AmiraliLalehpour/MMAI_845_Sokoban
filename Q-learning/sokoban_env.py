import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from room_utils import generate_room
from render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
from queue import Queue


class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=2,
                 num_gen_steps=None,
                 reset=True):
         
        self.initial_room_fixed = None
        self.initial_room_state = None
        self.initial_box_mapping = None  
        
        self.num_boxes = num_boxes
        self.boxes_on_target = 0
        
        self.box_movement_count = [0] * self.num_boxes
         
        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps



        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -20
        self.reward_box_on_target = 20
        self.reward_finished = 30
        self.reward_last = 0
        self.push_stuck_penalty = -1

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            # Initialize Room
            _ = self.reset()
   
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False
        
        if moved_box:
            box_index = self.room_state[self.new_box_position[0], self.new_box_position[1]] - 3
            self.box_movement_count[box_index] += 1

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
            
        if self.new_box_position is not None:
            x, y = self.new_box_position
            
            is_on_target = self.room_state[x, y] in [3]  
            
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            
            num_blocked = sum(1 for nx, ny in neighbors if self.room_state[nx, ny] in [0,3,4])
             
            if (self.room_state[x-1, y] in [0,3,4] and self.room_state[x+1, y] in [0,3,4]) \
               or (self.room_state[x, y-1] in [0,3,4] and self.room_state[x, y+1] in [0,3,4]):
                num_blocked -= 1
                
            if num_blocked >= 2 and not is_on_target:
                self.reward_last += self.push_stuck_penalty
            
            new_box_steps = self._calculate_steps_to_nearest_target(self.new_box_position)
            old_box_steps = self._calculate_steps_to_nearest_target(self.old_box_position)
            
            step_change = new_box_steps - old_box_steps
               
            total_step_change = self._calculate_total_step_change(self.new_box_position, self.old_box_position)
            
            if total_step_change < 0:
                self.reward_last += 0.2
            elif total_step_change > 0:
                self.reward_last -= 0.2
                
        # Check if there are any remaining boxes that are not on targets
        remaining_boxes = np.argwhere(self.room_state == 4)  # 4 对应 'box not on target'
        if remaining_boxes.size == 0:
            # All boxes are on targets, no penalty
            return

        # Check if any remaining box can be pushed to a target position
        can_reach_target = False
        for box_position in remaining_boxes:
            x, y = box_position[0], box_position[1]

            # Check if the box can be pushed to any neighboring target position
            for target_position in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                tx, ty = target_position
                if self.room_state[tx, ty] in [2]:  # 2 对应 'box target'
                    can_reach_target = True
                    break

            if can_reach_target:
                break

        # Apply penalty if there are remaining boxes that cannot be pushed to a target position
        if not can_reach_target:
            remaining_boxes_penalty = 0.5  
            self.reward_last -= remaining_boxes_penalty

        box_stagnation_penalty = 0.02  
        for box_index, movement_count in enumerate(self.box_movement_count):
            if movement_count == 0:  # Box has not been moved
                self.reward_last -= box_stagnation_penalty
            
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()
    
    def _calculate_steps_to_nearest_target(self, position):
        # Calculate the distance from the given position to the nearest target
        target_positions = np.argwhere(self.room_fixed == 2)  # Find all target positions

        if len(target_positions) == 0:
            return 0
        
        queue = Queue()
        visited = set()
        x, y = position
        queue.put((x, y, 0))
        visited.add((x, y))

        while not queue.empty():
            x, y, steps = queue.get()

            if (x, y) in target_positions:
                return steps

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if self._is_valid(nx, ny) and (nx, ny) not in visited:
                    queue.put((nx, ny, steps + 1))
                    visited.add((nx, ny))
        return 0
    def _is_valid(self, x, y):
        return 0 <= x < self.dim_room[0] and 0 <= y < self.dim_room[1] and self.room_state[x, y] not in [0, 3]
    
    def _calculate_total_step_change(self, new_box_position, old_box_position):
        total_step_change = 0
        
        for target_position in np.argwhere(self.room_fixed == 2):
            old_steps = self._calculate_steps_to_nearest_target(old_box_position)
            new_steps = self._calculate_steps_to_nearest_target(new_box_position)
            total_step_change += (new_steps - old_steps)

        return total_step_change

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, second_player=False, render_mode='rgb_array'):
        if self.initial_room_fixed is None:
            self.initial_room_fixed, self.initial_room_state, self.initial_box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
            
        self.room_fixed = np.copy(self.initial_room_fixed)
        self.room_state = np.copy(self.initial_room_state)
        self.box_mapping = np.copy(self.initial_box_mapping)

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

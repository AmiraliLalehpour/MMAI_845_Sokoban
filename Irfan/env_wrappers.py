import gym
import cv2
import numpy as np

class SokobanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SokobanGymWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, observation):
        # Convert RGB observation to grayscale
        gray_obs = np.dot(observation, [0.2989, 0.5870, 0.1140])
        # Resize to 84x84 using bilinear interpolation
        resized_obs = cv2.resize(gray_obs, (84, 84), interpolation=cv2.INTER_LINEAR)
        # Convert to uint8 to match the observation space
        return np.uint8(resized_obs)



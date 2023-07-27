#pip install stable-baselines3
#pip install 'shimmy>=0.2.1'
#pip install gymnasium

import numpy as np
import time
from stable_baselines3 import DQN
#from stable_baselines3.common.envs import SokobanEnv

def preprocess_observation(obs):
    """
    Preprocess the observation for DQN with CNN policy.
    Transpose the observation image to match the expected shape.
    """
    #obs = np.mean(obs, axis=-1, keepdims=True)  # Convert to grayscale
    #obs = np.transpose(obs, (2, 0, 1))  # Transpose (height, width, channels) to (channels, height, width)
    #obs = obs / 255.0  # Normalize to [0, 1]
    return obs


def dqn_CNN(env, num_episodes, batch_size, learning_rate, discount_factor, exploration_fraction, exploration_final_eps):
    # Create and train the DQN agent with CNN policy
    model = DQN('CnnPolicy', env, learning_rate=learning_rate, buffer_size=10000, exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps)

    model.learn(total_timesteps=num_episodes)

    # Evaluate the trained agent
    total_rewards = []
    for episode in range(10):  # Evaluate 10 episodes
        obs = env.reset_new()
        print("Observation shape before preprocessing:", obs.shape)
        env.render(mode='human')
        done = False
        total_reward = 0
        step=0

        while not done:
            obs = preprocess_observation(obs)
            print("Observation shape after preprocessing:", obs.shape)
            env.render(mode='human')
            time.sleep(0.01)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(int(action))
            total_reward += reward
            step += 1
            print("Episode: {}, Step: {}, Total Reward: {}".format(episode + 1, step, total_reward))



        total_rewards.append(total_reward)

    return np.mean(total_rewards)


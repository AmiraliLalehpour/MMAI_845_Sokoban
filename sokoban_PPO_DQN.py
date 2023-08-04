import numpy as np
import matplotlib as plt
import time
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

#Change this import to match the file name of the environment file
from sokoban_env import custom_sokoban_env

# May not need these imports
import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from gym.wrappers import Monitor
from room_utils import generate_room
from render_utils import room_to_rgb, room_to_tiny_world_rgb
##

# Create an instance of the custom_sokoban_env environment
env = custom_sokoban_env(dim_room=(10, 10), max_steps=120, num_boxes=3, num_gen_steps=30, reset=True)
#print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Action space:", env.action_space)

# If the environment don't follow the Stable baseline 3 interface, an error will be thrown
print("Check env: ")
check_env(env, warn=True)
obs = env.reset()


# Test the environment to see if it can do basic actions
def test_env(env, pause):
        GO_LEFT = 7
        # Hardcoded best agent: always go left!
        n_steps = 20
        for step in range(n_steps):
            print(f"Step {step + 1}")
            obs, reward, terminated, truncated = env.step(GO_LEFT)
            done = terminated or truncated
            print("obs=", obs, "reward=", reward, "done=", done)
            env.render()
            time.sleep(pause)
            if done:
                print("Goal reached!", "reward=", reward)
                break

test_env(env,pause=0.1)

# Instantiate the env by wrapping it into a Stablebaseline 3 compatible environment
vec_env = make_vec_env(custom_sokoban_env, n_envs=1, env_kwargs=dict(dim_room=(10, 10), max_steps=120, num_boxes=3, num_gen_steps=30, reset=True))

# Define and train the PPO model with a CNN policy
def train_model_ppo(vec_env):
        num_episodes = 200
        # Create the PPO model with a CNN policy
        # Hyperparameters
        n_steps = 128
        learning_rate = 0.03
        gamma = 0.99
        gae_lambda = 0.99
        ent_coef = 0.1
        vf_coef = 0.5
        max_grad_norm = 0.5
        #clip_range = 0.2
        batch_size = 64

        # Create the PPO model with a CNN policy
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            #clip_range=clip_range
        )
        # Train the agent for 20000 timesteps. Increase to as high as 250,000 if you have a GPU
        model.learn(total_timesteps=20000)
        # Save the trained model if needed to avoid having to repeatedly train
        model.save("ppo_sokoban")
        return model


# Define and train the DQN model with a CNN policy
def train_model_dqn(vec_env):
    # Create the DQN model with a CNN policy
    # Hyperparameters
    buffer_size = 10000
    learning_rate = 0.001
    exploration_fraction = 0.1
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.02
    batch_size = 64
    learning_starts = 1000
    target_update_interval = 1000
    train_freq = 4

    model = DQN(
        "CnnPolicy",
        vec_env,
        verbose=1,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        batch_size=batch_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        train_freq=train_freq
    )

    # Train the agent for 20000 timesteps. Increase to as high as 250,000 if you have a GPU
    model.learn(total_timesteps=20000)

    # Save the trained model if needed
    model.save("dqn_sokoban")
    return model

# Evaluate the trained agent
def test_agent_avg(model, vec_env):
        mean_reward = 0.0
        n_eval_episodes = 300
        episode_rewards_list = []

        for episode in range(n_eval_episodes):
            obs = vec_env.reset()
            done = False
            episode_reward = 0.0
            step = 0
            while not done:
                action, _ = model.predict(obs)
                vec_env.render()
                obs, reward, done, _ = vec_env.step(action)
                episode_reward += reward
                step += step
                print(f'Episode {episode + 1} - Step {step} - Reward: {reward}')
            mean_reward += episode_reward
            episode_rewards_list.append((episode + 1, episode_reward))
            #print(f'Episode {episode + 1} - Episode Reward: {episode_reward}')
            mean_reward /= n_eval_episodes
            print("Episodes reward", episode_rewards_list)
            print("Mean reward", mean_reward)
        return episode_rewards_list


# Plot the Cumulative rewards vs. the number of episodes
def plot_rew(title, rew_list, label):
    # Unpack the list of tuples (episode_number, episode_reward) into separate lists
    episode_numbers, episode_rewards = zip(*rew_list)
    #"Sokoban Warehouse:",
    plt.title("Sokoban Warehouse: " + title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(episode_numbers, episode_rewards, label=label)
    # Set the y-axis limits to -11 and 11
    plt.ylim(-20, 10)
    plt.legend()
    plt.show()

# Flags to run SARSA and Q-learning
run_PPO = False
run_DQN = True

if run_PPO:
    print("------ Running PPO-learning ------")
    model = train_model_ppo(vec_env)
    model = PPO.load('ppo_sokoban')
    episode_reward = test_agent_avg(model, vec_env)
    plot_rew("PPO Agent Performance", episode_reward, "PPO")

if run_DQN:
    print("------ Running DQN-learning ------")
    model = train_model_dqn(vec_env)
    model = DQN.load("dqn_sokoban")
    episode_reward = test_agent_avg(model, vec_env)
    plot_rew("DQN Agent Performance", episode_reward, "DQN")

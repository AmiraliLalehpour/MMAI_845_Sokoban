import tensorflow as tf
import numpy as np
import gym
import cv2

def build_dqn_model(input_shape, num_actions, learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    return model

def build_dqn_model(input_shape, num_actions, learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    return model


def preprocess_observation(obs):
    return obs


def train_dqn(env, num_episodes, learning_rate, discount_factor, exploration_fraction, exploration_final_eps, target_model_update):
    # Get observation and action space dimensions
    obs = env.reset()
    obs = preprocess_observation(obs)
    input_shape = obs.shape  # Update the input shape here
    num_actions = env.action_space.n

    # Build the DQN model
    model = build_dqn_model(input_shape, num_actions, learning_rate)

    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs = preprocess_observation(obs)
            obs = np.expand_dims(obs, axis=0)  # Add batch dimension
            action_values = model.predict(obs)
            action = np.argmax(action_values)
            print(obs.shape)

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            next_obs = preprocess_observation(next_obs)
            next_obs = np.expand_dims(next_obs, axis=0)  # Add batch dimension

            target = reward
            if not done:
                next_action_values = model.predict(next_obs)
                target += discount_factor * np.max(next_action_values)

            target_f = action_values.copy()
            target_f[0, action] = target

            # Train the model using the target
            #model.fit(obs, target_f, epochs=1, verbose=0)

            obs = next_obs

        total_rewards.append(total_reward)

    return np.mean(total_rewards)
import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

# Constants for Q-Learning
EPISODES = 30000            # Total number of training episodes
LEARNING_RATE = 0.1         # Learning rate (alpha) for updating Q-values
DISCOUNT_FACTOR = 0.99      # Discount factor (gamma) for future rewards
EPSILON = 1.0               # Initial epsilon for epsilon-greedy policy
EPSILON_DECAY = 0.999       # Decay rate for epsilon to balance exploration and exploitation

def default_Q_value():
    """Returns the default Q-value for uninitialized state-action pairs."""
    return 0

if __name__ == "__main__":
    # Initialize the CliffWalking environment
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)  # Set random seed for reproducibility

    # Initialize the Q-table with a default value of 0 for each state-action pair
    Q_table = defaultdict(default_Q_value)

    # Record rewards for the last 100 episodes to monitor training progress
    episode_reward_record = deque(maxlen=100)

    # Main loop for training over the specified number of episodes
    for i in range(EPISODES):
        episode_reward = 0      # Track the total reward for the current episode
        done = False            # Flag to indicate if the episode is finished
        obs = env.reset()[0]    # Reset the environment and get the initial observation

        # Q-Learning Implementation
        while not done:
            # Epsilon-greedy policy for action selection (exploration vs. exploitation)
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: choose a random action
            else:
                # Exploit: choose the action with the highest Q-value for the current state
                prediction = np.array([Q_table[(obs, a)] for a in range(env.action_space.n)])
                action = np.argmax(prediction)
            
            # Take the selected action and observe the next state, reward, and termination flag
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Update the done flag if the episode ends

            # Q-Learning update rule
            if not done:
                # Estimate the maximum future reward for the next state
                next_Q_values = [Q_table[(next_obs, a)] for a in range(env.action_space.n)]
                max_next_Q = max(next_Q_values)
                
                # Update the Q-value for the current state-action pair
                current_Q = Q_table[(obs, action)]
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * current_Q + \
                                         LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_Q)
            else:
                # If the episode ends, update Q-value without considering future rewards
                current_Q = Q_table[(obs, action)]
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * current_Q + LEARNING_RATE * reward

            # Move to the next state
            obs = next_obs
            episode_reward += reward  # Accumulate the reward for this episode

        # Decay epsilon to gradually reduce exploration over time
        EPSILON *= EPSILON_DECAY
            
        # Logging progress every 100 episodes
        if i % 100 == 0 and i > 0:
            avg_reward = sum(list(episode_reward_record)) / 100
            print(f"LAST 100 EPISODE AVERAGE REWARD: {avg_reward}")
            print(f"EPSILON: {EPSILON}")

        # Record the reward for the current episode
        episode_reward_record.append(episode_reward)
    
    # Save the trained Q-table and the final epsilon value
    with open('Q_TABLE_QLearning.pkl', 'wb') as model_file:
        pickle.dump([Q_table, EPSILON], model_file)

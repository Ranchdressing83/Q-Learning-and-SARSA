import gymnasium as gym
import pickle
import random
import numpy as np
import time

def default_Q_value():
    """Returns the default Q-value for uninitialized state-action pairs."""
    return 0

def evaluate_rl_agent(Q_table, EPSILON, env_name, visualize=False):
    """
    Evaluates a reinforcement learning agent over 100 episodes.

    Parameters:
        Q_table (dict): The Q-table containing state-action values.
        EPSILON (float): The exploration rate for epsilon-greedy policy.
        env_name (str): The name of the Gymnasium environment.
        visualize (bool): Whether to render the environment during evaluation.

    Returns:
        float: The average reward over 100 episodes.
    """
    total_reward = 0
    env = gym.envs.make(env_name)
    env.reset(seed=1)  # Set random seed for reproducibility

    # Run the agent for 100 episodes
    for i in range(100):
        obs = env.reset()[0]  # Get initial observation
        done = False          # Flag to track if the episode has ended

        while not done:
            # Epsilon-greedy action selection (explore or exploit)
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: random action
            else:
                prediction = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)       # Exploit: best action based on Q-table

            # Take the chosen action and observe the next state and reward
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward  # Accumulate reward

            # Render the environment if visualization is enabled
            if visualize:
                env.render()
                time.sleep(0.01)

    # Calculate the average reward over 100 episodes
    score = total_reward / 100
    return score

def test_RL_agent(config, visualize=False):
    """
    Tests a reinforcement learning agent using a saved Q-table.

    Parameters:
        config (tuple): A tuple containing the environment name and algorithm name.
        visualize (bool): Whether to render the environment during testing.
    """
    env_name, algo_name = config[0], config[1]

    # Load the saved Q-table and epsilon value from a file
    loaded_data = pickle.load(open(f'Q_TABLE_{algo_name}.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]

    # Evaluate the agent and print the average reward
    score = evaluate_rl_agent(Q_table, EPSILON, env_name, visualize=visualize)
    print(f"{algo_name} on {env_name}:")
    print(f"Average episode-reward over 100 episodes is {score}")

if __name__ == "__main__":
    # Configuration: environment name and algorithm name
    print('-' * 40)
    config = ('CliffWalking-v0', 'QLearning')
    # config = ('CliffWalking-v0', 'SARSA')  # Uncomment to test SARSA instead of Q-Learning

    try:
        # Test the reinforcement learning agent
        test_RL_agent(config)
    except Exception as e:
        # Handle and print any errors that occur during testing
        print(e)

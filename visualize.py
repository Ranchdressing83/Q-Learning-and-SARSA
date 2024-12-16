import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_policy(q_table, config):
    """
    Visualizes the optimal policy derived from the Q-table for the given environment.

    Parameters:
        q_table (dict): The Q-table containing state-action values.
        config (tuple): A tuple containing the environment name and algorithm name.
    """
    # Define the environment dimensions for CliffWalking-v0
    env_name, _ = config
    if env_name == 'CliffWalking-v0':
        n_rows, n_cols = 4, 12
        # Mapping of action indices to arrows for visualization
        action_to_arrow = {
            0: '↑',  # up
            1: '→',  # right
            2: '↓',  # down
            3: '←'   # left
        }
    
    # Create a grid to hold the best action (policy) for each state
    policy_grid = np.empty((n_rows, n_cols), dtype=object)
    
    # Iterate through each state and determine the best action based on Q-values
    for state in range(n_rows * n_cols):
        row = state // n_cols
        col = state % n_cols
        prediction = np.array([q_table[(state, i)] for i in range(4)])  # Q-values for all actions
        best_action = np.argmax(prediction)                             # Best action (highest Q-value)
        policy_grid[row, col] = action_to_arrow[best_action]           # Assign arrow to policy grid
    
    if env_name == 'CliffWalking-v0':
        # Create the plot for the policy visualization
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        
        # Add arrows representing the best actions to the plot
        for row in range(n_rows):
            for col in range(n_cols):
                ax.text(col + 0.5, n_rows - row - 0.5, policy_grid[row, col], 
                        ha='center', va='center', fontsize=20)
        
        # Highlight the cliff area (danger zone) in red
        cliff_area = plt.Rectangle((1, 0), 10, 1, fill=True, color='red', alpha=0.3)
        ax.add_patch(cliff_area)
        
        # Mark the start and goal positions
        ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(n_cols - 0.5, 0.5, 'G', ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Customize the plot title
        ax.set_title('Cliff Walking Policy')

        # Display the plot
        plt.show()

def default_Q_value():
    """Returns the default Q-value for uninitialized state-action pairs."""
    return 0

def test_RL_agent(config, visualize=False):
    """
    Loads a Q-table from a saved file and returns it for policy visualization.

    Parameters:
        config (tuple): A tuple containing the environment name and algorithm name.
        visualize (bool): Whether to visualize the agent's performance.

    Returns:
        dict: The loaded Q-table.
    """
    env_name, algo_name = config[0], config[1]

    # Load the saved Q-table and epsilon value from the corresponding file
    loaded_data = pickle.load(open(f'Q_TABLE_{algo_name}.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]
    return Q_table

if __name__ == "__main__":
    # Configuration: specify the environment and algorithm name
    print('-' * 40)
    config = ('CliffWalking-v0', 'QLearning')
    # Uncomment the following line to test SARSA instead of Q-Learning
    # config = ('CliffWalking-v0', 'SARSA')

    try:
        # Load the Q-table using the specified configuration
        Q_table = test_RL_agent(config)
    except Exception as e:
        print(e)

    # Visualize the policy derived from the Q-table
    plot_policy(Q_table, config)

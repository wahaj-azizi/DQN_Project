from env_ import create_env  # Assuming the class and function are in a file named home_food_env.py
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1  # Exploration rate
epsilon_min = 0.3  # Minimum exploration rate
epsilon_decay = 0.999   # Decay rate for exploration
no_episodes = 10_000  # Number of episodes

goal_coordinates = (9, 9)
hell_state_coordinates = [(5, 2), (2, 5)]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    env = create_env(goal_coordinates=goal_coordinates,
                     hell_state_coordinates=hell_state_coordinates)
                    #  ,reward_state_coordinates=reward_state_coordinates)

    # Train a Q-learning agent:
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy")

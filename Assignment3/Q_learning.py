import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):
    # Initialize the Q-table
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm
    for episode in range(no_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            # env.render(done)
            total_reward += reward

            # Debug prints to trace the values
            #print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

            # Update the Q-values using the Q-value update rule
            q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + alpha * \
                (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])

            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


def visualize_q_table(hell_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(9, 9),
                      #reward_state_coordinates=[(4, 8), (8, 3)],
                      actions=["Right", "Down", "Left", "Up"],
                      q_values_path="q_table.npy"):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for hell_state in hell_state_coordinates:
                mask[hell_state] = True
            # for reward_state in reward_state_coordinates:
            #     mask[reward_state] = True  # Don't mask reward states

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Mark Goal state
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            # Mark Hell states
            for hell_state in hell_state_coordinates:
                ax.text(hell_state[1] + 0.5, hell_state[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
            ax.invert_yaxis()  # Fix the inverted Y-axis

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
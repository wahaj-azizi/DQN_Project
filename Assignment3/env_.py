import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import random
import matplotlib.image as mpimg

class HomeFoodEnvStatic(gym.Env):
    def __init__(self, grid_size=10, max_steps=500):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_state = np.array([0, 0])
        self.goal_state = np.array([grid_size - 1, grid_size - 1])
        self.hell_states = [np.array([5, 2]), np.array([2, 5])]  

        self.agent_image_path = r"ASSIGNMENT_1_DATA/teenage.png"
        self.goal_image_path = r"ASSIGNMENT_1_DATA/biryani.jpg"
        self.biryani_image_path = r"ASSIGNMENT_1_DATA/biryani.jpg"
        self.salad_image_path = r"ASSIGNMENT_1_DATA/Salad.png"

        self.burger_image_path = r"ASSIGNMENT_1_DATA/burger.png"
        self.fries_image_path = r"ASSIGNMENT_1_DATA/fries.png"

        # Load images
        self.agent_image = mpimg.imread(self.agent_image_path)
        self.goal_image = mpimg.imread(self.goal_image_path)
        self.biryani_image = mpimg.imread(self.biryani_image_path)
        self.salad_image = mpimg.imread(self.salad_image_path)

        self.burger_image = mpimg.imread(self.burger_image_path)
        self.fries_image = mpimg.imread(self.fries_image_path)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)
        self.steps_taken = 0
        self.total_reward = 0.0
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')  
        plt.show(block=False)

    def reset(self):

        self.agent_state = np.array(random.choice([[0, 0], [0, 9], [9, 0]]))
        self.steps_taken = 0
        self.total_reward = 0.0
        return tuple(self.agent_state)   
    


    def step(self, action):
        self.steps_taken += 1
        if action == 0 and self.agent_state[1] < self.grid_size - 1:  # up
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:  # right
            self.agent_state[0] += 1

        reward = -0.1  # Small penalty for each step to encourage quicker finding of food
        done = np.array_equal(self.agent_state, self.goal_state)

        if done:
            reward = 10.0
        elif np.array_equal(self.agent_state, self.hell_states[0]):
            reward = -5.0  # Large penalty for entering the first hell state
        elif np.array_equal(self.agent_state, self.hell_states[1]):
            reward = -5.0  # Large penalty for entering the second hell state


        self.total_reward += reward

        if self.steps_taken >= self.max_steps:
            done = True

        info = {}
        return tuple(self.agent_state), reward, done, info


    def render(self, done):
        self.ax.clear()
        self.ax.set_facecolor('darkgray')  # Set the axes background to black

        # Draw hell states
        self.ax.imshow(self.fries_image, extent=(self.hell_states[0][0] - 0.5, self.hell_states[0][0] + 0.5,
                                                 self.hell_states[0][1] - 0.5, self.hell_states[0][1] + 0.5), aspect='auto')
        self.ax.imshow(self.burger_image, extent=(self.hell_states[1][0] - 0.5, self.hell_states[1][0] + 0.5,
                                                  self.hell_states[1][1] - 0.5, self.hell_states[1][1] + 0.5), aspect='auto')




        # Draw agent and goal
        self.ax.imshow(self.agent_image, extent=(self.agent_state[0] - 0.5, self.agent_state[0] + 0.5,
                                                 self.agent_state[1] - 0.5, self.agent_state[1] + 0.5), aspect='auto')
        self.ax.imshow(self.goal_image, extent=(self.goal_state[0] - 0.5, self.goal_state[0] + 0.5,
                                                self.goal_state[1] - 0.5, self.goal_state[1] + 0.5), aspect='auto')

        if done:
            if np.array_equal(self.agent_state, self.goal_state):
                self.ax.imshow(self.biryani_image, extent=(0, self.grid_size, 0, self.grid_size), aspect='auto')
            else:
                self.ax.imshow(self.salad_image, extent=(0, self.grid_size, 0, self.grid_size), aspect='auto')

        # Set limits and aspect ratio
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")

        # Display total reward
        self.ax.text(2.0, -1.5, f'Total Reward: {self.total_reward:.2f}', color='white', fontsize=12)#, ha='center')
        self.ax.text(4.5, -2.5, f'Looking for Pakistani cousine in Germany', color='white', fontsize=12, ha='center')

        plt.pause(0.1)

    def close(self):
        plt.close()

def create_env(goal_coordinates, hell_state_coordinates, grid_size=10, max_steps=500):#, reward_state_coordinates):
    """
    Function to create and configure the HomeFoodEnvStatic environment.

    Args:
    goal_coordinates (tuple): Coordinates of the goal state.
    hell_state_coordinates (list of tuples): List of coordinates for the hell states.
    reward_state_coordinates (list of tuples): List of coordinates for the reward states.
    grid_size (int, optional): Size of the grid. Defaults to 10.
    max_steps (int, optional): Maximum number of steps per episode. Defaults to 500.

    Returns:
    HomeFoodEnvStatic: Configured environment instance.
    """
    # Create the environment with the specified grid size and max steps
    env = HomeFoodEnvStatic(grid_size=grid_size, max_steps=max_steps)
    
    # Set the goal state
    env.goal_state = np.array(goal_coordinates)
    
    # Set the hell states
    env.hell_states = [np.array(coord) for coord in hell_state_coordinates]
    
    return env
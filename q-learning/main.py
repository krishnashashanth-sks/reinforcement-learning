import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from train import train
from evaluate import evaluate

# 1. Define the number of bins for each dimension
# CartPole observation space dimensions:
# 0: Cart Position (-4.8 to 4.8)
# 1: Cart Velocity (-Inf to Inf)
# 2: Pole Angle (-0.418 rad to 0.418 rad)
# 3: Pole Angular Velocity (-Inf to Inf)

# We'll use 10 bins for each dimension. For unbounded dimensions, we need to set practical limits.
# Gymnasium CartPole-v1 environment limits:
# Cart Position: -2.4 to 2.4 (episode ends if outside these)
# Pole Angle: -0.2095 rad to 0.2095 rad (episode ends if outside these)
# Velocity and Angular Velocity are theoretically unbounded but practically have limits observed during episodes.

# Let's set the bounds and number of bins for each dimension.
# For velocity and angular velocity, we'll use slightly larger bounds than typically observed
# to avoid issues with values falling outside bins during early exploration.

position_bins = np.linspace(-2.4, 2.4, 10)
velocity_bins = np.linspace(-3.0, 3.0, 10) # Approximated practical limits
angle_bins = np.linspace(-0.2095, 0.2095, 10)
angular_velocity_bins = np.linspace(-3.0, 3.0, 10) # Approximated practical limits

bins = [position_bins, velocity_bins, angle_bins, angular_velocity_bins]


# Re-create the CartPole environment to get action space size
env = gym.make('CartPole-v1')

# 1. Determine Q-table dimensions
# Each dimension's number of discrete states is len(bins[i])
state_space_size = [len(b) for b in bins]
action_space_size = env.action_space.n

# Initialize the Q-table with zeros
q_table_shape = tuple(state_space_size + [action_space_size])
q_table = np.zeros(q_table_shape)

print(f"Q-table shape: {q_table.shape}")

# 2. Define Hyperparameters
learning_rate = 0.1       # alpha
discount_factor = 0.99    # gamma
exploration_rate = 1.0    # epsilon (starts with full exploration)
min_exploration_rate = 0.01
exploration_decay_rate = 0.995
n_episodes = 20000

# Close the environment as it's no longer needed for just getting action_space.n
env.close()

env = gym.make('CartPole-v1')

rewards_per_episode = train(n_episodes,env,q_table,discount_factor,learning_rate,min_exploration_rate,exploration_decay_rate,bins)

# Set exploration rate to minimum for evaluation (or 0.0 for pure exploitation)
evaluation_exploration_rate = min_exploration_rate # Use the value set during training (0.01)

num_evaluation_episodes = 100 # Run for 100 evaluation episodes

env = gym.make('CartPole-v1')

rewards_evaluation_episodes=evaluate(num_evaluation_episodes,env,q_table,bins)

# Plotting total reward per episode
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, label='Reward per Episode')
plt.title('Total Reward Per Training Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot rolling average
window_size = 100
rolling_average = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(rolling_average, label=f'Rolling Average (Last {window_size} Episodes)')
plt.title('Q-Learning Training Progress: Rolling Average of Rewards')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training rolling average and evaluation rewards for comparison
plt.figure(figsize=(12, 6))
plt.plot(rolling_average, label=f'Training Rolling Average (Last {window_size} Episodes)', color='blue')
plt.axhline(y=np.mean(rewards_evaluation_episodes), color='red', linestyle='--', label=f'Average Evaluation Reward ({len(rewards_evaluation_episodes)} Episodes: {np.mean(rewards_evaluation_episodes):.2f})')
plt.title('Q-Learning Performance: Training vs. Evaluation')
plt.xlabel('Episode (Training)')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

print("Plots generated to visualize training and evaluation progress.")
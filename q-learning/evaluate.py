import numpy as np
from utils import discretize_state

def evaluate(num_evaluation_episodes,env,q_table,bins):
    rewards_evaluation_episodes = []

    print(f"\nStarting evaluation of the trained Q-Learning agent for {num_evaluation_episodes} episodes...")

    for episode in range(num_evaluation_episodes):
        observation, info = env.reset()
        current_state = discretize_state(observation, bins)

        terminated = False
        truncated = False
        total_reward_episode = 0

        while not terminated and not truncated:
            # Always exploit the learned policy (greedy action)
            # Using argmax directly from the q_table
            action = np.argmax(q_table[current_state])

            new_observation, reward, terminated, truncated, info = env.step(action)
            new_state = discretize_state(new_observation, bins)

            total_reward_episode += reward
            current_state = new_state

        rewards_evaluation_episodes.append(total_reward_episode)

        if (episode + 1) % 10 == 0:
            print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward_episode}")

    average_evaluation_reward = np.mean(rewards_evaluation_episodes)
    print(f"\nEvaluation complete. Average reward over {num_evaluation_episodes} episodes: {average_evaluation_reward:.2f}")
    env.close()
    return rewards_evaluation_episodes
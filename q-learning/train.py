import numpy as np
from utils import discretize_state

def train(n_episodes,env,q_table,discount_factor,learning_rate,min_exploration_rate,exploration_decay_rate,bins):
    rewards_per_episode = []

    print("Starting Q-Learning training...")

    for episode in range(n_episodes):
        observation, info = env.reset()
        current_state = discretize_state(observation, bins)

        terminated = False
        truncated = False
        total_reward_episode = 0

        while not terminated and not truncated:
            # Epsilon-greedy policy
            if np.random.uniform(0, 1) > exploration_rate: # Exploit
                action = np.argmax(q_table[current_state])
            else: # Explore
                action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)
            new_state = discretize_state(new_observation, bins)

            # Q-Learning update rule
            q_table[current_state][action] = q_table[current_state][action] + learning_rate * \
                                            (reward + discount_factor * np.max(q_table[new_state]) - q_table[current_state][action])

            total_reward_episode += reward
            current_state = new_state

        rewards_per_episode.append(total_reward_episode)

        # Exploration rate decay
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay_rate)

        if (episode + 1) % 1000 == 0:
            average_reward_last_100 = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}: Average reward over last 100 episodes: {average_reward_last_100:.2f}, Exploration Rate: {exploration_rate:.4f}")

    env.close()
    print("Training complete.")
    return rewards_per_episode
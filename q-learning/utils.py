import numpy as np

def discretize_state(observation, bins):
    discretized_state = []
    for i, obs_val in enumerate(observation):
        # Use np.digitize to map the observation value to its bin index
        # np.digitize returns indices such that bins[i-1] <= x < bins[i]
        # We subtract 1 to get 0-indexed bin numbers, and clip to handle edge cases
        bin_index = np.digitize(obs_val, bins[i]) - 1
        # Ensure the bin index is within the valid range [0, num_bins-1]
        bin_index = np.clip(bin_index, 0, len(bins[i]) - 1)
        discretized_state.append(int(bin_index))
    return tuple(discretized_state)
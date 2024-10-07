import numpy as np


def custom_discounted_rewards(rewards, gamma=0.99, alpha=0.0):
    """
    Calculate the custom discounted rewards with an optional alpha parameter.

    Args:
        rewards (list or np.array): List of rewards at each timestep.
        gamma (float): Standard discount factor (0 <= gamma <= 1).
        alpha (float): A parameter to adjust the weight on future rewards.

    Returns:
        np.array: Array of discounted rewards.
    """
    T = len(rewards)
    discounted_rewards = np.zeros(T)

    # Iterating from the last reward to the first
    for t in range(T):
        discount_sum = 0.0
        for k in range(T - t):
            # Apply the custom discounting function
            discount_factor = (gamma**k) * (1 + alpha * k)
            discount_sum += discount_factor * rewards[t + k]
        discounted_rewards[t] = discount_sum

    return discounted_rewards


# Example usage
rewards = [1 for i in range(32)]  # Example rewards at each time step
gamma = 0.99  # Discount factor
alpha = 0.1  # Linear adjustment for the past rewards

# discounted = custom_discounted_rewards(rewards, gamma, alpha)
# print(discounted)


def apply_outcome_discount(step_reward_list, compound_rate=0.99):
    """use before normalization"""
    result_list = []
    counter = 1
    for i in step_reward_list[::-1]:
        result_list.append(i * (compound_rate**counter))
        counter += 1
    return result_list[::-1]


print(apply_outcome_discount(rewards, compound_rate=0.995))

import random


def select_version_by_ratio(versions):
    epsilon = 1e-10
    ratios = [version["ratio"] for version in versions]

    if not abs(sum(ratios) - 1.0) <= epsilon:
        raise ValueError(f"Sum of ratios must be 1.0, now {sum(ratios)}")

    cumulative_ratios = []
    cumulative_sum = 0
    for ratio in ratios:
        cumulative_sum += ratio
        cumulative_ratios.append(cumulative_sum)

    random_value = random.random()
    for idx, cumulative_ratio in enumerate(cumulative_ratios):
        if random_value <= cumulative_ratio:
            return versions[idx]

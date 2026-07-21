def survival_probabilities(num_blocks: int, p_L: float = 0.5) -> list[float]:
    """Linear decay rule from the paper: p_l = 1 - (l / L) * (1 - p_L), l = 1..L."""
    return [1 - (l / num_blocks) * (1 - p_L) for l in range(1, num_blocks + 1)]

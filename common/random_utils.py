import os
import random
from typing import List

DEFAULT_SEED = 42


def get_seed() -> int:
    seed_env = os.getenv("SEED")
    if seed_env:
        try:
            return int(seed_env)
        except ValueError:
            print(f"Invalid SEED env: {seed_env} => Using default seed: {DEFAULT_SEED}")
            return DEFAULT_SEED
    else:
        return DEFAULT_SEED


def get_unique_permutations(n: int, num_of_permutations: int) -> List[List[int]]:
    """
    Get `num_of_permutations` unique random permutations from `list(range(n))`
    """
    permutation_set = set()
    while len(permutation_set) < num_of_permutations:
        permutation = random.sample(range(n), n)
        permutation_set.add(tuple(permutation))

    return [list(permutation) for permutation in permutation_set]

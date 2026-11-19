
import random
import time

import numpy as np

from naive import naive_multiply
from divide_conquer import divide_conquer
from strassen import strassen

# strassen.py switches to the naive triple loop once a block is this small.
# Recursing all the way to 1x1 has so much overhead that it never beats naive.
LEAF_SIZE = 64

# divide_conquer recurses down to 2x2, so it is too slow to run on big inputs.
DC_MAX_SIZE = 128


def random_matrix(n, low=-9, high=9, seed=None):
    rng = random.Random(seed)
    return [[rng.randint(low, high) for _ in range(n)] for _ in range(n)]


# Example matrices: (name, A, B)
EXAMPLES = [
    (
        "3x3 example",
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
    ),
    (
        "4x4 example",
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]],
    ),
    ("5x5 identity", np.eye(5, dtype=int).tolist(), random_matrix(5, seed=1)),
    ("16x16 random", random_matrix(16, seed=2), random_matrix(16, seed=3)),
    ("33x33 random (odd)", random_matrix(33, seed=4), random_matrix(33, seed=5)),
    ("64x64 random", random_matrix(64, seed=6), random_matrix(64, seed=7)),
    ("128x128 random", random_matrix(128, seed=8), random_matrix(128, seed=9)),
    # Large enough that Strassen beats naive
    ("256x256 random", random_matrix(256, seed=10), random_matrix(256, seed=11)),
    ("512x512 random", random_matrix(512, seed=12), random_matrix(512, seed=13)),
]

ALGORITHMS = [
    ("Naive", naive_multiply),
    ("Divide & Conquer", divide_conquer),
    ("Strassen", lambda A, B: strassen(A, B, leaf_size=LEAF_SIZE)),
]


def time_it(func, A, B):
    start = time.perf_counter()
    result = func(A, B)
    elapsed = time.perf_counter() - start
    return np.asarray(result), elapsed


def main():
    for name, A, B in EXAMPLES:
        n = len(A)
        print(f"\n=== {name} ===")
        expected = np.asarray(A) @ np.asarray(B)

        if n <= 4:
            print("A =")
            print(np.asarray(A))
            print("B =")
            print(np.asarray(B))
            print("A * B =")
            print(expected)

        for alg_name, func in ALGORITHMS:
            if func is divide_conquer and n > DC_MAX_SIZE:
                print(f"  {alg_name:<18} {'skipped':>10}      (too slow above {DC_MAX_SIZE}x{DC_MAX_SIZE})")
                continue
            result, elapsed = time_it(func, A, B)
            ok = "correct" if np.array_equal(result, expected) else "WRONG"
            print(f"  {alg_name:<18} {elapsed * 1000:>10.3f} ms   ({ok})")


if __name__ == "__main__":
    main()

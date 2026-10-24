"""Strassen's matrix multiplication.

Matrices are represented as lists of lists (row-major). The main entry point
is `strassen(A, B)`, which multiplies two square n x n matrices using 7
recursive multiplications per level instead of the usual 8, giving
O(n^log2(7)) ~ O(n^2.81) time.

The small helpers below (add, subtract, split, join, pad) are minimal
versions so this file runs on its own; they can be swapped for the shared
helper module once it exists.
"""


# --- Minimal helpers ---------------------------------------------------------

def _add(X, Y):
    return [[x + y for x, y in zip(rx, ry)] for rx, ry in zip(X, Y)]


def _sub(X, Y):
    return [[x - y for x, y in zip(rx, ry)] for rx, ry in zip(X, Y)]

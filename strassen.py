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


def _split(M):
    """Split an even-sized square matrix into four quadrants."""
    h = len(M) // 2
    return (
        [row[:h] for row in M[:h]],
        [row[h:] for row in M[:h]],
        [row[:h] for row in M[h:]],
        [row[h:] for row in M[h:]],
    )


def _join(C11, C12, C21, C22):
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom


def _naive_multiply(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _next_power_of_two(n):
    size = 1
    while size < n:
        size *= 2
    return size

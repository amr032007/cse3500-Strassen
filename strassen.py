"""Strassen's matrix multiplication.

Matrices are represented as lists of lists (row-major). The main entry point
is `strassen(A, B)`, which multiplies two square n x n matrices using 7
recursive multiplications per level instead of the usual 8, giving
O(n^log2(7)) ~ O(n^2.81) time.

The small helpers below (add, subtract, split, join, pad) are minimal
versions so this file runs on its own; they can be swapped for the shared
helper module once it exists.
"""


def strassen(A, B, leaf_size=1):
    """Return the product A * B of two square matrices using Strassen's algorithm.

    leaf_size: below or at this size, fall back to the naive triple loop.
    Matrices whose size is not a power of two are zero-padded and the result
    is trimmed back to n x n.
    """
    n = len(A)
    if n == 0:
        return []
    if len(B) != n or any(len(row) != n for row in A) or any(len(row) != n for row in B):
        raise ValueError("strassen() requires two square matrices of the same size")

    size = _next_power_of_two(n)
    if size != n:
        A = _pad(A, size)
        B = _pad(B, size)

    C = _strassen_recursive(A, B, max(1, leaf_size))
    return [row[:n] for row in C[:n]]


def _strassen_recursive(A, B, leaf_size):
    n = len(A)
    if n <= leaf_size:
        return _naive_multiply(A, B)

    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    # The 7 products
    M1 = _strassen_recursive(_add(A11, A22), _add(B11, B22), leaf_size)
    M2 = _strassen_recursive(_add(A21, A22), B11, leaf_size)
    M3 = _strassen_recursive(A11, _sub(B12, B22), leaf_size)
    M4 = _strassen_recursive(A22, _sub(B21, B11), leaf_size)
    M5 = _strassen_recursive(_add(A11, A12), B22, leaf_size)
    M6 = _strassen_recursive(_sub(A21, A11), _add(B11, B12), leaf_size)
    M7 = _strassen_recursive(_sub(A12, A22), _add(B21, B22), leaf_size)

    # Combine into the four quadrants of C
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_add(_sub(M1, M2), M3), M6)

    return _join(C11, C12, C21, C22)


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


def _pad(M, size):
    n = len(M)
    padded = [row + [0] * (size - n) for row in M]
    padded += [[0] * size for _ in range(size - n)]
    return padded

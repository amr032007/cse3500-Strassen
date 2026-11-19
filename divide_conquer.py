import numpy as np


def divide_conquer(A, B):
    
    A = np.asarray(A)
    B = np.asarray(B)
    n = len(A)

    if n <= 2:  # Base case
        return np.dot(A, B)

    # Odd size: pad with a zero row/column, multiply, then trim back to n x n
    if n % 2 == 1:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        return divide_conquer(A, B)[:n, :n]

    # Partition matrices into submatrices
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # 8 recursive multiplications
    C11 = divide_conquer(A11, B11) + divide_conquer(A12, B21)
    C12 = divide_conquer(A11, B12) + divide_conquer(A12, B22)
    C21 = divide_conquer(A21, B11) + divide_conquer(A22, B21)
    C22 = divide_conquer(A21, B12) + divide_conquer(A22, B22)

    # Combine quadrants to form C
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C


# Example usage:
if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    print(divide_conquer(A, B))

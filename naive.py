def naive_multiply(A, B):
    
    n = len(A)
    m = len(B)
    p = len(B[0])
    if len(A[0]) != m:
        raise ValueError("Inner dimensions do not match")

    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            total = 0
            for k in range(m):
                total += A[i][k] * B[k][j]
            C[i][j] = total
    return C


# Example usage:
if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    print(naive_multiply(A, B))

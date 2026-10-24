# cse3500-Strassen

Implementations of Strassen's matrix multiplication algorithm for CSE 3500.

Strassen's algorithm multiplies two n x n matrices by splitting each into four
n/2 x n/2 quadrants and computing the result with **7** recursive
multiplications instead of the usual 8. This gives a running time of
O(n^log2(7)) ≈ O(n^2.81), compared to O(n^3) for regular multiplication.

## Running

```
python strassen.py
python strassen_v2.py
```

Both files multiply the same two 3x3 example matrices and print the same
result (`strassen_v2.py` shows it in numpy's array format):

```
[[30, 24, 18], [84, 69, 54], [138, 114, 90]]
```

`strassen_v2.py` requires numpy (`pip install numpy`). `strassen.py` has no
dependencies.

## strassen.py (pure Python)

Matrices are plain lists of lists.

- `strassen(A, B, leaf_size=1)` is the entry point. It checks that both
  matrices are square and the same size. If the size is not a power of two, it
  pads both matrices with zeros up to the next power of two, runs the
  algorithm, and trims the result back to n x n.
- `_strassen_recursive(A, B, leaf_size)` does the actual work: it splits A and
  B into quadrants, computes the 7 products M1-M7, and combines them into the
  four quadrants of the result:
  - C11 = M1 + M4 - M5 + M7
  - C12 = M3 + M5
  - C21 = M2 + M4
  - C22 = M1 - M2 + M3 + M6
- Once a submatrix is `leaf_size` or smaller, it falls back to regular
  triple-loop multiplication. Raising `leaf_size` is useful for comparing
  speed, since recursing all the way down to 1x1 has a lot of overhead.
- Small helpers handle adding, subtracting, splitting, joining, and padding
  matrices.

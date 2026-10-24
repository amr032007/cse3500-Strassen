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

## strassen_v2.py (numpy)

Uses numpy arrays, so adding, subtracting, and slicing out quadrants are
single operations (`A[:mid, :mid]`, `A + B`).

- Converts the inputs to numpy arrays, so lists of lists work too.
- Base case: matrices of size 2 or smaller are multiplied directly with
  `np.dot`.
- Odd sizes: pads one zero row and column to make the size even, recurses, and
  trims the extra row and column off the result. This happens at each level
  where the size is odd, rather than padding to a power of two up front.
- Computes the 7 products P1-P7 and combines them:
  - C11 = P5 + P4 - P2 + P6
  - C12 = P1 + P2
  - C21 = P3 + P4
  - C22 = P5 + P1 - P3 - P7
- Joins the quadrants back together with `np.vstack` / `np.hstack`.

## How these were written

- **strassen.py** was generated with Claude (an AI assistant) as a working
  reference implementation.
- **strassen_v2.py** is my own implementation of the algorithm. Claude helped
  fix two bugs in it: converting list inputs to numpy arrays so the 2D slicing
  works, and padding odd-sized matrices so the quadrants split evenly.

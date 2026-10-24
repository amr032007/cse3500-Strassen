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

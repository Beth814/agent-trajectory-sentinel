# Short-horizon diagnosis: washout and CUSUM slack

**SUPPORTING / diagnostic evidence, on development data.** Every row refits the whole per-deployment calibration at that setting, on the same healthy fold, at the same 5% budget. Shipped setting is washout 3, k 0.5. Nothing here selects a value: a setting chosen because it maximises a column of this table would be tuned on the corpus it is measured on.

## autogen

| washout | k | theta | detection | healthy FA | injected episodes scoring 0 |
|---|---|---|---|---|---|
| 0 | 0.0 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.1 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.25 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.5 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 1 | 0.0 | 39.41 | 0.783 | 0.083 | 0 |
| 1 | 0.1 | 39.11 | 0.783 | 0.083 | 0 |
| 1 | 0.25 | 38.66 | 0.783 | 0.083 | 1 |
| 1 | 0.5 | 37.91 | 0.783 | 0.083 | 2 |
| 2 | 0.0 | 33.58 | 0.870 | 0.083 | 0 |
| 2 | 0.1 | 33.48 | 0.870 | 0.083 | 1 |
| 2 | 0.25 | 33.33 | 0.870 | 0.083 | 2 |
| 2 | 0.5 | 33.08 | 0.870 | 0.083 | 2 |
| 3 | 0.0 | 9.68 | 0.913 | 0.125 | 2 |
| 3 | 0.1 | 8.88 | 0.913 | 0.125 | 2 |
| 3 | 0.25 | 7.72 | 0.913 | 0.125 | 2 |
| 3 | 0.5 | 5.97 | 0.913 | 0.125 | 2 |

## langgraph

| washout | k | theta | detection | healthy FA | injected episodes scoring 0 |
|---|---|---|---|---|---|
| 0 | 0.0 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.1 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.25 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.5 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 1 | 0.0 | 280.99 | 0.667 | 0.042 | 0 |
| 1 | 0.1 | 280.39 | 0.667 | 0.042 | 0 |
| 1 | 0.25 | 279.49 | 0.667 | 0.042 | 0 |
| 1 | 0.5 | 277.99 | 0.667 | 0.042 | 0 |
| 2 | 0.0 | 282.03 | 0.667 | 0.042 | 0 |
| 2 | 0.1 | 281.53 | 0.667 | 0.042 | 0 |
| 2 | 0.25 | 280.78 | 0.667 | 0.042 | 0 |
| 2 | 0.5 | 279.53 | 0.667 | 0.042 | 0 |
| 3 | 0.0 | 193.86 | 0.625 | 0.000 | 0 |
| 3 | 0.1 | 193.46 | 0.625 | 0.000 | 0 |
| 3 | 0.25 | 192.86 | 0.625 | 0.000 | 0 |
| 3 | 0.5 | 191.86 | 0.625 | 0.000 | 0 |

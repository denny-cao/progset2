import numpy as np
import sys

def split(x):
    """
    Split an n x n matrix x into four submatrices, handling both even and odd n.
    """
    rows, cols = x.shape
    row_half = (rows + 1) // 2  # Adjusted to handle odd rows
    col_half = (cols + 1) // 2  # Adjusted to handle odd cols

    A = x[:row_half, :col_half]
    B = x[:row_half, col_half:]
    C = x[row_half:, :col_half]
    D = x[row_half:, col_half:]

    return A, B, C, D

def strassen(x, y, n_0=27):
    """
    Strassen's algorithm for matrix multiplication, with modifications to handle matrices of all sizes.
    """
    if x.shape[0] <= n_0 or x.shape[1] <= n_0 or y.shape[0] <= n_0 or y.shape[1] <= n_0:
        return standard(x, y)

    # Ensure matrices are square and of even dimension by padding if necessary
    max_dim = max(x.shape[0], x.shape[1], y.shape[0], y.shape[1])
    if max_dim % 2 != 0:
        max_dim += 1  # Increase dimension to make it even
    x_padded = np.pad(x, ((0, max_dim - x.shape[0]), (0, max_dim - x.shape[1])))
    y_padded = np.pad(y, ((0, max_dim - y.shape[0]), (0, max_dim - y.shape[1])))

    A, B, C, D = split(x_padded)
    E, F, G, H = split(y_padded)

    P1 = strassen(A, F - H)
    P2 = strassen(A + B, H)
    P3 = strassen(C + D, E)
    P4 = strassen(D, G - E)
    P5 = strassen(A + D, E + H)
    P6 = strassen(B - D, G + H)
    P7 = strassen(A - C, E + F)

    result1 = P5 + P4 - P2 + P6
    result2 = P1 + P2
    result3 = P3 + P4
    result4 = P1 + P5 - P3 - P7

    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    result = np.vstack((top, bottom))

    return result[:x.shape[0], :y.shape[1]]

def standard(x, y):
    """
    Standard matrix multiplication algorithm, unaltered.
    """
    result = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            for k in range(x.shape[1]): 
                result[i, j] += x[i, k] * y[k, j]
    return result


def main():


if __name__ == "__main__":
    main()

# Task 1: Assume that the cost of any single arithmetic operation (adding, subtracting, multiplying, or dividing two real
# numbers) is 1, and that all other operations are free. Consider the following variant of Strassen's algorithm: to
# multiply two n by n matrices, start using Strassen's algorithm, but stop the recursion at some size n_0, and use the
# conventional algorithm below that point. You have to find a suitable value for n_0 --- the cross-over point.
# Analytically determine the value of n_0 that optimizes the running time of this algorithm in this model (That is,
# solve the appropriate equations, somehow, numerically.) This gives a crude estimate for the cross-over point between
# Strassen's algorithm and the standard matrix multiplication algorithm

# Task 2: Implement your variant of Strassen's algorithm and the standard matrix multiplication algorithm to find the
# corss-over point experimentally. Experimentally optimize for n_0 and compare the experimental results with your
# estimate from above. Make both implementations as efficient as possible. The actual cross-over point, which you would
# like to make as small as possible, will depend on how efficiently you implement Strassne's algorithm. Your
# implementation should work on any size matrices, not just those whose dimensions are a power of 2. To test your
# algorithm, you might try matrices where each entry is randomly selected to be 0 or 1; similarly, you might try
# matrices where each entry is randomly selected to be 0, 1, or 2, or instead 0, 1, or -1. We will test on integer
# matrices, possibly of this form (You may assume integer inputs.) You need not try all of these, but do test your
# algorithm adequately. 

import numpy as np

def split(x):
    """
    Split an n x n matrix x to four n/2 x n/2 submatrices, handling both even and odd n (A is the top left, B is the
    top right, C is the bottom left, and D is the bottom right)
    """

    n = x.shape[0]
    if n % 2 == 0:
        half = n // 2
    
        A = x[:half, :half]
        B = x[:half, half:]
        C = x[half:, :half]
        D = x[half:, half:]
    
        return A, B, C, D
    else:
        x = np.pad(x, ((0, 1), (0, 1)))
        return split(x)

def strassen(x, y, n_0=1):
    """
    Strassen's algorithm for matrix multiplication
    """

    # Base case (stop recursion at crossover point n_0 and use standard algorithm)
    if x.shape[0] <= n_0:
        return standard(x, y)

    # Split matrices into submatrices
    A, B, C, D = split(x)
    E, F, G, H = split(y)

    # Recursively compute products
    P1 = strassen(A, F - H, n_0)
    P2 = strassen(A + B, H, n_0)
    P3 = strassen(C + D, E, n_0)
    P4 = strassen(D, G - E, n_0)
    P5 = strassen(A + D, E + H, n_0)
    P6 = strassen(B - D, G + H, n_0)
    P7 = strassen(A - C, E + F, n_0)

    # Compute submatrices of the results
    result1 = -P2 + P4 + P5 + P6
    result2 = P1 + P2
    result3 = P3 + P4
    result4 = P1 - P3 + P5 - P7

    # Combine submatrices into the results
    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    return np.vstack((top, bottom))

def standard(x, y):
    """
    Standard matrix multiplication algorithm
    """

    # Create matrix of zeroes
    result = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[0]):
                result[i, j] += x[i, k] * y[k, j]

    return result

def winograd(x, y, n_0=1):
    """
    Winograd's algorithm for matrix multiplication.
    """

    if len(x) <= n_0:
        return standard(x, y)

    # Split matrices into submatrices
    a, b, c, d = split(x)
    A, C, B, D = split(y)

    t = winograd(a,A, n_0)
    u = winograd((c-a), (C-D), n_0)
    v = winograd((c+d), (C-A), n_0)
    w = t + winograd((c + d - a), (A+D-C), n_0)

    result1 = t + winograd(b, B, n_0)
    result2 = w + v + winograd((a + b - c - d),D, n_0)
    result3 = w + u + winograd(d, (B + C - A - D), n_0)
    result4 = w + u + v

    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    return np.vstack((top, bottom))

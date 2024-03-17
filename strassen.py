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

# Task 3: Triangle in random graphs: Recall that you can represent the adjacency matrix of a graph by a matrix A.
# Consider an undirected graph. It turns out that A^3 can be used to determine the number of triangles in a gram: the
# (ij)th entry in the matrix A^2 counts the paths from i to j of length two, and the (ij)th entry in the matrix A^3
# counts the path from i to j of length 3. To count the number of triangles in a graph, we can simply add the entries
# in the diagonal, and divide by 6. This is because the jth diagonal entry counts the number of paths of length 3 from
# j to j. Each such path is a triangle, and each triangle is counted 6 times (for each of the vertices in the triangle,
# it is counted once in each direction).

# Create a random graph on 1024 vertices where each edge is included with probability p for each of the following
# values fo p: p = 0.01, 0.02, 0.03, 0.04, and 0.05. Use your (Strassen's) matrix multiplication code to count the
# number of triangels in each of the graphs, and compare it to the expected number of triangles, which is
# \comb{1024}{3}p^3. Create a chart showing your results compared to the expectation.

import numpy as np

def split(x):
    """
    Split an n x n matrix x to four n/2 x n/2 submatrices, handling both even and odd n (A is the top left, B is the
    top right, C is the bottom left, and D is the bottom right)
    """
    n = x.shape[0]
    half = n // 2

    A = x[:half, :half]
    B = x[:half, half:]
    C = x[half:, :half]
    D = x[half:, half:]

    return A, B, C, D

def strassen(x, y):
    """
    Strassen's algorithm for matrix multiplication
    """

    # Base case (n=1)
    if len(x) == 1:
        return x * y

    # Split matrices into submatrices
    A, B, C, D = split(x)
    E, F, G, H = split(y)

    # Recursively compute products
    P1 = strassen(A, F - H)
    P2 = strassen(A + B, H)
    P3 = strassen(C + D, E)
    P4 = strassen(D, G - E)
    P5 = strassen(A + D, E + H)
    P6 = strassen(B - D, G + H)
    P7 = strassen(A - C, E + F)

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

    # Iterate through rows of x
    for i in range(x.shape[0]):
        # Iterate through columns of y
        for j in range(y.shape[1]):
            # Iterate through rows of y
            for k in range(y.shape[0]):
                result[i, j] += x[i, k] * y[k, j]

    return result

def main():
    # Test Strassen's algorithm
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])

    print(strassen(x, y))

    # Test standard matrix multiplication algorithm
    print(standard(x, y))

if __name__ == "__main__":
    main()

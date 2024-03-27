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
import time
import matplotlib.pyplot as plt


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

def pad(x):
    """
    Pad an n x n matrix x with zeroes to make n a power of 2. Used when initial matrix for Strassen's algorithm is not
    even.
    """

    n = 2 ** np.ceil(np.log2(x.shape[0])).astype(int)
    y = np.zeros((n, n), dtype=int)
    y[:x.shape[0], :x.shape[1]] = x
    return y

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

def measure_time(func, *args):
    """
    Measure the time taken to run a function with arguments
    """
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# def main():
# 
#     # Test Strassen's algorithm
#     matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     n_0_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# 
#     execution_times = {n_0: {'Strassen': [], 'Old': []} for n_0 in n_0_values}
# 
#     for n_0 in n_0_values:
#         for size in matrix_sizes:
#             x = np.random.randint(0, 3, (size, size))
#             y = np.random.randint(0, 3, (size, size))
# 
#             #account for odd sized matrix
#             original_size = size
#             if size % 2 != 0:
#                 x = pad(x)
#                 y = pad(y)
#                 size += 1
# 
#             _, time_strassen = measure_time(strassen, x, y, n_0)
#             if size == original_size:
#                 _, time_standard = measure_time(old_strassen, x, y)
#             else:
#                 _, time_standard = measure_time(old_strassen, x[:original_size, :original_size], y[:original_size, :original_size])
# 
#             execution_times[n_0]['Strassen'].append(time_strassen)
#             execution_times[n_0]['Old'].append(time_standard)
# 
#     print(execution_times)
# 
#     for n_0, times in execution_times.items():
#         plt.figure(figsize=(10, 6))
# 
#         plt.plot(matrix_sizes, times['Strassen'], label='Crossover', marker='o')
#         plt.plot(matrix_sizes, times['Old'], label='No Crossover', marker='x')
# 
#         # Add labels and title
#         plt.xlabel('Matrix size')
#         plt.ylabel('Execution time (s)')
#         plt.title(f'Execution time for n_0={n_0}')
#         plt.legend()
#         plt.grid(True, which='both', ls='--')
#         plt.yscale('log')
#         plt.xscale('log', base=2)
# 
#         plt.show()

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

def main():
    # Make x and y a random 23x23 matrix_sizes
    x = np.random.randint(1, 3, (23, 23))
    y = np.random.randint(1, 3, (23, 23))
    print("x")
    print(x)
    print("y")
    print(y)
    original_size = x.shape[0]
    if original_size % 2 != 0:
        x = pad(x)
        y = pad(y)

        print("Standard")
        print(standard(x, y)[:original_size, :original_size])
        print("Strassen")
        print(strassen(x, y)[:original_size, :original_size])
        print("Winograd")
        print(winograd(x, y)[:original_size, :original_size])
    else:
        print("Standard")
        print(standard(x, y))
        print("Strassen")
        print(strassen(x, y))
        print("Winograd")
        print(winograd(x, y))

if __name__ == "__main__":
    main()

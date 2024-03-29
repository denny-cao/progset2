import numpy as np
import sys

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

def strassen(x, y, n_0=27):
    """
    Strassen's algorithm for matrix multiplication
    """

    x_org = x.shape[0]
    y_org = y.shape[1]

    # Base case (stop recursion at crossover point n_0 and use standard algorithm)
    if x.shape[0] <= n_0:
        return standard(x, y)

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

    strassen_matrix = np.vstack((top, bottom))

    return strassen_matrix[:x_org, :y_org]

def standard(x, y):
    """
    Standard matrix multiplication algorithm
    """

    # Create matrix of zeroes
    result = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                result[i, j] += x[i, k] * y[k, j]

    return result

def winograd(x, y, n_0=1):
    """
    Winograd's algorithm for matrix multiplication. From Wikipedia: https://en.wikipedia.org/wiki/Strassen_algorithm
    """

    if len(x) <= n_0:
        return standard(x, y)

    # Split matrices into submatrices
    a, b, c, d = split(x)
    A, C, B, D = split(y)

    t = winograd(a,A)
    u = winograd((c-a), (C-D))
    v = winograd((c+d), (C-A))
    w = t + winograd((c + d - a), (A+D-C))

    result1 = t + winograd(b, B)
    result2 = w + v + winograd((a + b - c - d),D)
    result3 = w + u + winograd(d, (B + C - A - D))
    result4 = w + u + v

    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    return np.vstack((top, bottom))

def standard_optimized(x, y):
    """
    Optimize by changing order of loops ==> improve spatial locality (Almurayh 2022)
    """
    result = np.zeros_like(x)

    for i in range(x.shape[0]):
        for k in range(x.shape[0]):
            for j in range(x.shape[0]):
                result[i, j] += x[i, k] * y[k, j]

    return result

def main():
    if len(sys.argv) != 4:
        print("Usage: python strassen.py <flag> <dim> <input_file>")
        sys.exit(1)
    if int(sys.argv[1]) == 0: 
        dim = int(sys.argv[2])
        input_file = sys.argv[3]
    
        # Read input matrices
        with open(input_file, 'r') as f:
            # Read matrices from file
            A = np.zeros((dim, dim))
            B = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    A[i, j] = int(f.readline())
            for i in range(dim):
                for j in range(dim):
                    B[i, j] = int(f.readline())
    
            # Compute product using Strassen's algorithm, removing padding if necessary
            result = strassen(A, B)[:dim, :dim]
            # Print diagonal elements of the result matrix
            for i in range(dim):
                print(int(result[i, i]))
    
if __name__ == "__main__":
    main()

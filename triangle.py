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

import strassen
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def create_graph(n, p):
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

def count_triangles(G):
    A = np.dot(G, G)
    A = np.dot(A, G)
    return np.trace(A)/6

def expected_count(n, p):
    return math.comb(n, 3) * p**3

def create_chart(n, ps, triangles):
    plt.plot(ps, triangles, label='Count of triangles')
    plt.plot(ps, [expected_count(n, p) for p in ps], label='Expected count of triangles')
    plt.xlabel('Probability p')
    plt.ylabel('Count of triangles')
    plt.title('Count of triangles in random graphs')
    plt.legend()
    plt.show()

def main():
    n = 1024
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    triangles = []
    for p in ps:
        G = create_graph(n, p)
        triangles.append(count_triangles(G))
    create_chart(n, ps, triangles)

if __name__ == '__main__':
    main()

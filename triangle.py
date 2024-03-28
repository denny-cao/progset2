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
    A = strassen.strassen(G, G)
    A = strassen.strassen(A, G)
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

def create_table(n, ps, triangles):
    print("Graph Probability $p$ & Count of triangles & Expected count of triangles \\\\")
    print("\hline")
    for i in range(len(ps)):
        print(f"{ps[i]} & {triangles[i]} & {expected_count(n, ps[i])} \\\\")
    print("\hline")

def main():
    n = 1024
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    triangles = []
    for p in ps:
        G = create_graph(n, p)
        triangles.append(count_triangles(G))
    create_table(n, ps, triangles)
    create_chart(n, ps, triangles)

def average_trials_main():
    n = 1024
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    trials = 10
    triangles = []
    for p in ps:
        total = 0
        for _ in range(trials):
            G = create_graph(n, p)
            total += count_triangles(G)
        triangles.append(total/trials)
    create_table(n, ps, triangles)

if __name__ == '__main__':
    average_trials_main()

from strassen import strassen
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

def single_trial_table():
    n = 1024
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    triangles = []
    for p in ps:
        G = create_graph(n, p)
        expected = expected_count(n, p)
        triangle = count_triangles(G)
        triangles.append(triangle)

    create_table(n, ps, triangles)

def average_trials_graph():
    n = 1024
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    trials = 5

    # Create dictionary to store separate trials
    results = {}
    for i in range(trials):
        results[i] = []

    for p in ps:
        for i in range(trials):
            G = create_graph(n, p)
            triangle = count_triangles(G)
            results[i].append(triangle)

    # Calculate average of trials
    averages = [0] * len(ps)
    for i in range(trials):
        for j in range(len(results[i])):
            averages[j] += results[i][j]
    averages = [average/trials for average in averages]

    expected = [expected_count(n, p) for p in ps]

    # Table of all trials and then a row for averages
    for i in range(trials):
        print(f"Trial {i+1}: {results[i]}")

    print(f"Averages: {averages}")
    print(f"Expected: {expected}")

if __name__ == '__main__':
    average_trials_graph()

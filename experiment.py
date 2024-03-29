from strassen import strassen, winograd, standard, split, standard_optimized
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

NUM_TRIALS = 5

def strassen_test(x, y):
    """
    Strassen's algorithm for matrix multiplication with only 1 call to strassen to find the optimal crossover point
    """

    # Base case (stop recursion at crossover point n_0 and use standard algorithm)
    if x.shape[0] <= 1:
        return standard(x, y)

    # Split matrices into submatrices
    A, B, C, D = split(x)
    E, F, G, H = split(y)

    # Recursively compute products
    P1 = standard(A, F - H)
    P2 = standard(A + B, H)
    P3 = standard(C + D, E)
    P4 = standard(D, G - E)
    P5 = standard(A + D, E + H)
    P6 = standard(B - D, G + H)
    P7 = standard(A - C, E + F)

    # Compute submatrices of the results
    result1 = -P2 + P4 + P5 + P6
    result2 = P1 + P2
    result3 = P3 + P4
    result4 = P1 - P3 + P5 - P7

    # Combine submatrices into the results
    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    return np.vstack((top, bottom))

def winograd_test(x, y):
    """
    Winograd's algorithm for matrix multiplication.
    """

    if len(x) <= 1:
        return standard(x, y)

    # Split matrices into submatrices
    a, b, c, d = split(x)
    A, C, B, D = split(y)

    t = standard(a,A)
    u = standard((c-a), (C-D))
    v = standard((c+d), (C-A))
    w = t + standard((c + d - a), (A+D-C))

    result1 = t + standard(b, B)
    result2 = w + v + standard((a + b - c - d),D)
    result3 = w + u + standard(d, (B + C - A - D))
    result4 = w + u + v

    top = np.hstack((result1, result2))
    bottom = np.hstack((result3, result4))

    return np.vstack((top, bottom))

def measure_time(func, *args):
    """
    Measure the time taken to run a function with arguments
    """

    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start


def experiment():
    """
    Run an experiment to find the optimal crossover point for Strassen's algorithm with input matrices of power of 2
    """

    # Test Strassen's algorithm
    matrix_sizes = [512,513]
    n_0_values = [x for x in range(5, 130)]

    execution_times = {n_0: {'Strassen w/Crossover': [], 'Winograd w/Crossover': []} for n_0 in n_0_values}
    baseline_times = {'Standard': [], 'Strassen w/o Crossover': [], 'Winograd w/o Crossover': []}

    for size in matrix_sizes:
        x = np.random.randint(0, 1, (size, size))
        y = np.random.randint(0, 1, (size, size))

        # Measure time for standard algorithm
        _, time_standard = measure_time(standard, x, y)
        baseline_times['Standard'].append(time_standard)
        _, time_strassenwo = measure_time(strassen, x, y, 1)
        baseline_times['Strassen w/o Crossover'].append(time_strassenwo)
        _, time_winogradwo = measure_time(winograd, x, y, 1)
        baseline_times['Winograd w/o Crossover'].append(time_winogradwo)

        for n_0 in n_0_values:
            _, time_strassen = measure_time(strassen, x, y, n_0)
            _, time_winograd = measure_time(winograd, x, y, n_0)
            execution_times[n_0]['Strassen w/Crossover'].append(time_strassen)
            execution_times[n_0]['Winograd w/Crossover'].append(time_winograd)

    # Plot the results for separate n_0 values
    for n_0 in n_0_values:
        plt.plot(matrix_sizes, execution_times[n_0]['Strassen w/Crossover'], label=f'Strassen w/Crossover n_0={n_0}')
        plt.plot(matrix_sizes, execution_times[n_0]['Winograd w/Crossover'], label=f'Winograd w/Crossover n_0={n_0}')
        plt.plot(matrix_sizes, baseline_times['Standard'], label='Standard')
        plt.plot(matrix_sizes, baseline_times['Strassen w/o Crossover'], label='Strassen w/o Crossover')
        plt.plot(matrix_sizes, baseline_times['Winograd w/o Crossover'], label='Winograd w/o Crossover')
        plt.xlabel('Matrix Size')
        plt.ylabel('Execution Time (s)')
        plt.title('Comparison of Standard, Strassen, and Winograd Algorithms')
        plt.legend()
        plt.savefig(f'plots/{n_0}.png')
        plt.clf()

    # Create csv file 
#     with open('experiment.csv', 'w') as f:
#         f.write('Matrix Size,Standard,Strassen w/o Crossover,Winograd w/o Crossover')
#         for n_0 in n_0_values:
#             f.write(f',Strassen w/Crossover n_0={n_0},Winograd w/Crossover n_0={n_0}')
#         f.write('\n')
#         for i in range(len(matrix_sizes)):
#             f.write(f'{matrix_sizes[i]},{baseline_times["Standard"][i]},{baseline_times["Strassen w/o Crossover"][i]},{baseline_times["Winograd w/o Crossover"][i]}')
#             for n_0 in n_0_values:
#                 f.write(f',{execution_times[n_0]["Strassen w/Crossover"][i]},{execution_times[n_0]["Winograd w/Crossover"][i]}')
#             f.write('\n')
    with open('experiment.csv', 'w', newline='') as f:
        fieldnames = ['Algorithm', 'Matrix Size', 'Execution Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(matrix_sizes)):
            writer.writerow({'Algorithm': 'Standard', 'Matrix Size': matrix_sizes[i], 'Execution Time': baseline_times['Standard'][i]})
            writer.writerow({'Algorithm': 'Strassen w/o Crossover', 'Matrix Size': matrix_sizes[i], 'Execution Time': baseline_times['Strassen w/o Crossover'][i]})
            writer.writerow({'Algorithm': 'Winograd w/o Crossover', 'Matrix Size': matrix_sizes[i], 'Execution Time': baseline_times['Winograd w/o Crossover'][i]})
            for n_0 in n_0_values:
                writer.writerow({'Algorithm': f'Strassen w/Crossover n_0={n_0}', 'Matrix Size': matrix_sizes[i], 'Execution Time': execution_times[n_0]['Strassen w/Crossover'][i]})
                writer.writerow({'Algorithm': f'Winograd w/Crossover n_0={n_0}', 'Matrix Size': matrix_sizes[i], 'Execution Time': execution_times[n_0]['Winograd w/Crossover'][i]})

        # Get n_0 value with minimum execution time with n_0 for each matrix size
        for i in range(len(matrix_sizes)):
            min_strassen = min([execution_times[n_0]['Strassen w/Crossover'][i] for n_0 in n_0_values])
            min_winograd = min([execution_times[n_0]['Winograd w/Crossover'][i] for n_0 in n_0_values])

            min_strassen_n_0 = n_0_values[np.argmin([execution_times[n_0]['Strassen w/Crossover'][i] for n_0 in n_0_values])]
            min_winograd_n_0 = n_0_values[np.argmin([execution_times[n_0]['Winograd w/Crossover'][i] for n_0 in n_0_values])]

            writer.writerow({'Algorithm': f'Strassen w/Crossover n_0={min_strassen_n_0}', 'Matrix Size': matrix_sizes[i], 'Execution Time': min_strassen})
            writer.writerow({'Algorithm': f'Winograd w/Crossover n_0={min_winograd_n_0}', 'Matrix Size': matrix_sizes[i], 'Execution Time': min_winograd})
    print('Experiment completed!')

def experiment2():
    """
    Test strassen w/o crossover for many matrix sizes
    """

    # Test Strassen's algorithm
    matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    execution_times_strassen = []
    execution_times_standard = []

    for size in matrix_sizes:
        x = np.random.randint(0, 1, (size, size))
        y = np.random.randint(0, 1, (size, size))

        _, time_strassen = measure_time(strassen, x, y, 1)
        _, time_standard = measure_time(standard, x, y)
        execution_times_strassen.append(time_strassen)
        execution_times_standard.append(time_standard)

    # Create CSV with data
    with open('experiment2.csv', 'w', newline='') as f:
        fieldnames = ['Matrix Size', 'Strassen w/o Crossover', 'Standard']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(matrix_sizes)):
            writer.writerow({'Matrix Size': matrix_sizes[i], 'Strassen w/o Crossover': execution_times_strassen[i], 'Standard': execution_times_standard[i]})

    # Graph the results using log scale for x axis

    plt.plot(matrix_sizes, execution_times_strassen, label='Strassen w/o Crossover')
    plt.plot(matrix_sizes, execution_times_standard, label='Standard')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Standard and Strassen Algorithms')
    plt.legend()
    plt.xscale('log')
    plt.savefig('plots/experiment2.png')
    plt.clf()
    print('Experiment completed!')


def test():
    """
    Test the correctness of the Strassen's algorithm
    """

    # Test Strassen's algorithm
    matrix_sizes = [8,16,31,32,62,124]


    for size in matrix_sizes:
        x = np.random.randint(5, 10, (size, size))
        y = np.random.randint(5, 10, (size, size))

        result_standard = standard(x, y)
        result_strassen = strassen(x, y)[:size, :size]
        # result_winograd = winograd(x, y, 1)
        print(result_standard)
        print(result_strassen)
        assert np.allclose(result_standard, result_strassen)
        print(f'Test passed for size {size}!')
        # assert np.allclose(result_standard, result_winograd[:x.shape[0], :x.shape[1]])

    print('All tests passed!')

def experiment3():
    """
    Run strassen_test and compare with standard for multiple inputs n to find the optimal crossover point
    """

    # Test Strassen's algorithm
    matrix_sizes = [x for x in range(1,101)]

    # Plot execution times for different matrix sizes

    execution_times_strassen = []
    execution_times_standard = []

    print('Matrix Size & Average Time Strassen & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(0, 1, (size, size))
            y = np.random.randint(0, 1, (size, size))

            _, time_strassen = measure_time(strassen_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_strassen += time_strassen
            avg_time_standard += time_standard

        execution_times_strassen.append(avg_time_strassen / NUM_TRIALS)
        execution_times_standard.append(avg_time_standard / NUM_TRIALS)
        
        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

    # Create CSV with data
    with open('experiment3.csv', 'w', newline='') as f:
        fieldnames = ['Matrix Size', 'Strassen w/Crossover', 'Standard']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(matrix_sizes)):
            writer.writerow({'Matrix Size': matrix_sizes[i], 'Strassen w/Crossover': execution_times_strassen[i], 'Standard': execution_times_standard[i]})

    # Graph the results using log scale for x axis

    plt.plot(matrix_sizes, execution_times_strassen, label='Strassen w/Crossover')
    plt.plot(matrix_sizes, execution_times_standard, label='Standard')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Standard and Strassen Algorithms')
    plt.legend()
    plt.xscale('log')
    plt.savefig('plots/experiment3.png')
    plt.clf()

    print('Experiment completed!')

def experiment4():
    """
    Run winograd_test and compare with standard for multiple inputs n to find the optimal crossover point
    """

    # Test Strassen's algorithm
    matrix_sizes = [x for x in range(1,101)]

    # Plot execution times for different matrix sizes

    execution_times_winograd = []
    execution_times_standard = []

    print('Matrix Size & Average Time Winograd & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_winograd = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(0, 1, (size, size))
            y = np.random.randint(0, 1, (size, size))

            _, time_winograd = measure_time(winograd_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_winograd += time_winograd
            avg_time_standard += time_standard

        execution_times_winograd.append(avg_time_winograd / NUM_TRIALS)
        execution_times_standard.append(avg_time_standard / NUM_TRIALS)
        
        print(f'{size} & {avg_time_winograd / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

    # Create CSV with data
    with open('experiment4.csv', 'w', newline='') as f:
        fieldnames = ['Matrix Size', 'Winograd w/Crossover', 'Standard']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(matrix_sizes)):
            writer.writerow({'Matrix Size': matrix_sizes[i], 'Winograd w/Crossover': execution_times_winograd[i], 'Standard': execution_times_standard[i]})

    # Graph the results using log scale for x axis

    plt.plot(matrix_sizes, execution_times_winograd, label='Winograd w/Crossover')
    plt.plot(matrix_sizes, execution_times_standard, label='Standard')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Standard and Winograd Algorithms')
    plt.legend()
    plt.xscale('log')
    plt.savefig('plots/experiment4.png')
    plt.clf()

    print('Experiment completed!')

def test2():
    """
    Test with matrix size 65
    """

    n = 128

    x = np.random.randint(0, 1, (n, n))
    y = np.random.randint(0, 1, (n, n))

    # Time for strassen and original

    _, time_strassen = measure_time(strassen, x, y, 1)
    _, time_standard = measure_time(standard, x, y)

    print(f'Strassen: {time_strassen}')
    print(f'Standard: {time_standard}')

def experiment5():
    """
    See what happens with different matrix entries
    """

    # Test Strassen's algorithm
    matrix_sizes = [x for x in range(1,101)]

    # Plot execution times for different matrix sizes

    execution_times_strassen = []
    execution_times_standard = []

    print('Matrix Size & Average Time Strassen & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(2**30, 2**31, (size, size))
            y = np.random.randint(2**30, 2**31, (size, size))

            _, time_strassen = measure_time(strassen_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_strassen += time_strassen
            avg_time_standard += time_standard

        execution_times_strassen.append(avg_time_strassen / NUM_TRIALS)
        execution_times_standard.append(avg_time_standard / NUM_TRIALS)
        
        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

    # Create CSV with data
    with open('experiment3.csv', 'w', newline='') as f:
        fieldnames = ['Matrix Size', 'Strassen w/Crossover', 'Standard']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(matrix_sizes)):
            writer.writerow({'Matrix Size': matrix_sizes[i], 'Strassen w/Crossover': execution_times_strassen[i], 'Standard': execution_times_standard[i]})

    # Graph the results using log scale for x axis

    plt.plot(matrix_sizes, execution_times_strassen, label='Strassen w/Crossover')
    plt.plot(matrix_sizes, execution_times_standard, label='Standard')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Standard and Strassen Algorithms')
    plt.legend()
    plt.xscale('log')
    plt.savefig('plots/experiment3.png')
    plt.clf()

    print('Experiment completed!')

def experiment6():
    """
    Generate tables for 0,1,2 matrices, -1,1 matrices and large random matrices
    """

    matrix_sizes = [x for x in range(1,51)]

    print('0,1,2 matrices')
    print('Matrix Size & Average Time Strassen & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(0, 3, (size, size))
            y = np.random.randint(0, 3, (size, size))

            _, time_strassen = measure_time(strassen_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_strassen += time_strassen
            avg_time_standard += time_standard

        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

    print('-1,1 matrices')
    print('Matrix Size & Average Time Strassen & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(-1, 2, (size, size))
            y = np.random.randint(-1, 2, (size, size))

            _, time_strassen = measure_time(strassen_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_strassen += time_strassen
            avg_time_standard += time_standard

        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

    print('Random matrices')
    print('Matrix Size & Average Time Strassen & Average Time Standard \\\\')
    print('\\hline')
    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(2**25, 2**26, (size, size))
            y = np.random.randint(2**25, 2**26, (size, size))

            _, time_strassen = measure_time(strassen_test, x, y)
            _, time_standard = measure_time(standard, x, y)

            avg_time_strassen += time_strassen
            avg_time_standard += time_standard

        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

def opt_stand_v_regular():
    matrix_sizes = [x for x in range(1,101)]

    print('Optimized Standard')
    print('Matrix Size & Average Time Optimized Standard & Average Time Standard \\\\')
    print('\\hline')

    for size in matrix_sizes:
        avg_time_strassen = 0
        avg_time_standard = 0
        for i in range(NUM_TRIALS):
            x = np.random.randint(0, 1, (size, size))
            y = np.random.randint(0, 1, (size, size))

            _, time_standard = measure_time(standard, x, y)
            _, time_optimized = measure_time(standard_optimized, x, y)

            avg_time_standard += time_standard
            avg_time_strassen += time_optimized

        print(f'{size} & {avg_time_strassen / NUM_TRIALS} & {avg_time_standard / NUM_TRIALS} \\\\')

if __name__ == '__main__':
    test()

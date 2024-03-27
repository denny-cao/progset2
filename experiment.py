from strassen import strassen, winograd, standard
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

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
    matrix_sizes = [3,9,27,81, 2, 4, 8, 16, 32, 64]

    for size in matrix_sizes:
        x = np.random.randint(0, 1, (size, size))
        y = np.random.randint(0, 1, (size, size))

        result_standard = standard(x, y)
        result_strassen = strassen(x, y, 1)
        result_winograd = winograd(x, y, 1)

        assert np.allclose(result_standard, result_strassen[:x.shape[0], :x.shape[1]])
        assert np.allclose(result_standard, result_winograd[:x.shape[0], :x.shape[1]])

    print('All tests passed!')

if __name__ == '__main__':
    experiment2()

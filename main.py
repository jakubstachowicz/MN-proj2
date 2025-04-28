import copy
import time
import numpy as np
import matplotlib.pyplot as plt

index = 198302
c = (index % 100) // 10
d = index % 10
e = (index % 1000) // 100
f = (index % 10000) // 1000
N_global = 1200 + 10 * c + d
norm_threshold = 1e-9
max_iterations = 200
chart_number = 1
num_of_runs_for_average = 10


def get_band_matrix_A(N, a1, a2, a3):
    main_diagonal = np.ones(N) * a1
    next_diagonal = np.ones(N - 1) * a2
    last_diagonal = np.ones(N - 2) * a3
    return (np.diag(main_diagonal)
            + np.diag(next_diagonal, 1)
            + np.diag(next_diagonal, -1)
            + np.diag(last_diagonal, 2)
            + np.diag(last_diagonal, -2))


# Exercise A
def get_matrix_A_vector_b(N=N_global, a1=5 + e, a2=-1, a3=-1):
    matrix_A = get_band_matrix_A(N, a1, a2, a3)
    vector_b = np.array([np.sin(n * (f + 1)) for n in range(1, N + 1)])
    return matrix_A, vector_b


# Exercise B part 1/2
def solve_Jacobi(N=N_global, a1=5 + e, a2=-1, a3=-1, show_chart=True):
    A, b = get_matrix_A_vector_b(N, a1, a2, a3)

    time_start = time.time()

    iteration_count = 0

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))

    M = -np.linalg.solve(D, L + U)
    w = np.linalg.solve(D, b)
    x = np.ones(N)
    r_norm = []

    inorm = np.linalg.norm(A @ x - b)
    r_norm.append(inorm)

    global norm_threshold, max_iterations
    while inorm > norm_threshold and iteration_count < max_iterations:
        x = M @ x + w
        inorm = np.linalg.norm(A @ x - b)
        iteration_count += 1
        r_norm.append(inorm)

    time_end = time.time()
    print(f"Solve Jacobi ended with {iteration_count} iterations in {time_end - time_start} seconds.")
    print(f"Norm at the end: {inorm}\n")

    if show_chart:
        global chart_number
        plt.semilogy(r_norm)
        plt.title(f"Wykres {chart_number}: Norma residuum w zależności od iteracji (Metoda Jacobiego)")
        chart_number += 1
        plt.xlabel("Iteracja")
        plt.ylabel("Rozmiar normy")
        plt.grid(True)
        plt.show()

    return time_end - time_start


# Exercise B part 2/2
def solve_Gauss_Seidel(N=N_global, a1=5 + e, a2=-1, a3=-1, show_chart=True):
    A, b = get_matrix_A_vector_b(N, a1, a2, a3)

    time_start = time.time()

    iteration_count = 0

    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    D = np.diag(np.diag(A))

    T = D + L
    w = np.linalg.solve(T, b)

    x = np.ones(N)
    r_norm = []

    inorm = np.linalg.norm(A @ x - b)
    r_norm.append(inorm)

    global norm_threshold, max_iterations
    while inorm > norm_threshold and iteration_count < max_iterations:
        x = -np.linalg.solve(T, U @ x) + w
        inorm = np.linalg.norm(A @ x - b)
        iteration_count += 1
        r_norm.append(inorm)

    time_end = time.time()
    print(f"Solve Gauss-Seidel ended with {iteration_count} iterations in {time_end - time_start} seconds.")
    print(f"Norm at the end: {inorm}\n")

    if show_chart:
        global chart_number
        plt.semilogy(r_norm)
        plt.title(f"Wykres {chart_number}: Norma residuum w zależności od iteracji (Metoda Gaussa-Seidla)")
        chart_number += 1
        plt.xlabel("Iteracja")
        plt.ylabel("Rozmiar normy")
        plt.grid(True)
        plt.show()

    return time_end - time_start


def LU_decomposition(A, m):
    U = copy.copy(A)
    L = np.eye(m)
    for i in range(2, m + 1):
        for j in range(1, i):
            L[i - 1, j - 1] = U[i - 1, j - 1] / U[j - 1, j - 1]
            U[i - 1, :] = U[i - 1, :] - L[i - 1, j - 1] * U[j - 1, :]
    return L, U


# Exercise D
def solve_direct(N=N_global, a1=5 + e, a2=-1, a3=-1):
    A, b = get_matrix_A_vector_b(N, a1, a2, a3)

    time_start = time.time()

    L, U = LU_decomposition(A, N)

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)

    r_norm = np.linalg.norm(A @ x - b)

    time_end = time.time()
    print(f"Solve direct ended in {time_end - time_start} seconds.")
    print(f"Norm at the end: {r_norm}\n")

    return time_end - time_start


# Exercise E
def compare_methods():
    matrix_sizes = [100, 500, 1000, 2000, 3000]

    time_direct = [0 for _ in range(len(matrix_sizes))]
    time_Jacobi = [0 for _ in range(len(matrix_sizes))]
    time_Gauss_Seidel = [0 for _ in range(len(matrix_sizes))]

    for idx, N in enumerate(matrix_sizes):
        print(f"=========================")
        print(f"Matrix size {N}")
        for i in range(num_of_runs_for_average):
            print(f"--- Run {i} ---")
            time_direct[idx] += solve_direct(N=N)
            time_Jacobi[idx] += solve_Jacobi(N=N, show_chart=False)
            time_Gauss_Seidel[idx] += solve_Gauss_Seidel(N=N, show_chart=False)
        time_direct[idx] /= num_of_runs_for_average
        time_Jacobi[idx] /= num_of_runs_for_average
        time_Gauss_Seidel[idx] /= num_of_runs_for_average

    global chart_number
    plt.semilogy(matrix_sizes, time_direct)
    plt.semilogy(matrix_sizes, time_Jacobi)
    plt.semilogy(matrix_sizes, time_Gauss_Seidel)
    plt.legend(['Metoda bezpośrednia (LU)', 'Metoda Jacobiego', 'Metoda Gaussa-Seidla'])
    plt.title(f"Wykres {chart_number}: Średni czas działania a metoda obliczeń")
    chart_number += 1
    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas, s")
    plt.grid(True)
    plt.show()

    plt.plot(matrix_sizes, time_direct)
    plt.plot(matrix_sizes, time_Jacobi)
    plt.plot(matrix_sizes, time_Gauss_Seidel)
    plt.legend(['Metoda bezpośrednia (LU)', 'Metoda Jacobiego', 'Metoda Gaussa-Seidla'])
    plt.title(f"Wykres {chart_number}: Średni czas działania a metoda obliczeń")
    chart_number += 1
    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas, s")
    plt.grid(True)
    plt.show()
    pass


if __name__ == '__main__':
    print("Parameters:")
    print("index = " + str(index))
    print("c = " + str(c))
    print("d = " + str(d))
    print("e = " + str(e))
    print("f = " + str(f))
    print("N_global = " + str(N_global))
    print("norm_threshold = " + str(norm_threshold))
    print("max_iterations = " + str(max_iterations))
    print()
    # Exercise B
    solve_Jacobi()
    solve_Gauss_Seidel()
    # Exercise C
    solve_Jacobi(a1=3)
    solve_Gauss_Seidel(a1=3)
    # Exercise D
    solve_direct(a1=3)
    # Exercise E
    compare_methods()

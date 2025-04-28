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
def solve_Jacobi(N=N_global, a1=5 + e, a2=-1, a3=-1):
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

    global chart_number
    plt.semilogy(r_norm)
    plt.title(f"Wykres {chart_number}: Norma residuum w zależności od iteracji (Metoda Jacobiego)")
    chart_number += 1
    plt.xlabel("Iteracja")
    plt.ylabel("Rozmiar normy")
    plt.grid(True)
    plt.show()


# Exercise B part 2/2
def solve_Gauss_Seidel(N=N_global, a1=5 + e, a2=-1, a3=-1):
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

    global chart_number
    plt.semilogy(r_norm)
    plt.title(f"Wykres {chart_number}: Norma residuum w zależności od iteracji (Metoda Gaussa-Seidla)")
    chart_number += 1
    plt.xlabel("Iteracja")
    plt.ylabel("Rozmiar normy")
    plt.grid(True)
    plt.show()


def LU_decomposition(A, m):
    U = copy.copy(A)
    L = np.eye(m)
    for i in range(2, m + 1):
        for j in range(1, i):
            L[i - 1, j - 1] = U[i - 1, j - 1] / U[j - 1, j - 1]
            U[i - 1, :] = U[i - 1, :] - L[i - 1, j - 1] * U[j - 1, :]
    return L, U


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

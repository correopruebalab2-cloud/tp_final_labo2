import numpy as np
from .sistema import armar_sistema
from .solvers import jacobi, gauss_seidel, jacobi_vectorizado

def avanzar(p, k, rhoCp, T0, solver='jacobi_vec'):
    """
    Loop temporal implícito (Euler atrás).
    solver: 'jacobi_vec' (default, rápido), 'jacobi' (bucles), 'gs' (Gauss-Seidel bucles)
    Devuelve: historia de T (array), tiempos, iteraciones por paso.
    """
    T = T0.copy()
    T_hist = [T.copy()]
    tiempos = [0.0]
    iters = []

    n_pasos = int(p.t_max / p.dt)
    for n in range(1, n_pasos + 1):
        t = n * p.dt
        A, b = armar_sistema(p, k, rhoCp, T, t)
        x0 = T.flatten()
        if solver == 'gs':
            x, it, err = gauss_seidel(A, b, x0, p.tol, p.max_iter)
        elif solver == 'jacobi':
            x, it, err = jacobi(A, b, x0, p.tol, p.max_iter)
        else:
            x, it, err = jacobi_vectorizado(A, b, x0, p.tol, p.max_iter)
        T = x.reshape(p.Nx, p.Ny)
        T_hist.append(T.copy())
        tiempos.append(t)
        iters.append(it)
        if n % 50 == 0:
            print("Paso", n, "t =", t, "s  Tmax =", np.max(T), "  iters =", it, "  err =", err)

    return np.array(T_hist), np.array(tiempos), np.array(iters)

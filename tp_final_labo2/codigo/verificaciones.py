import numpy as np
from .sistema import armar_sistema
from .solvers import jacobi_vectorizado

def _avanzar_con_S(p, k, rhoCp, T0, S, n_pasos, h_rob, dirichlet=None):
    """Loop temporal simple con fuente S fija. Devuelve T final."""
    h_orig = p.h_rob
    dt_orig = p.dt
    p.h_rob = h_rob
    T = T0.copy()
    for n in range(1, n_pasos + 1):
        A, b = armar_sistema(p, k, rhoCp, T, n * p.dt, S=S, dirichlet=dirichlet)
        x, it, err = jacobi_vectorizado(A, b, T.flatten(), p.tol, p.max_iter)
        T = x.reshape(p.Nx, p.Ny)
        if dirichlet is not None:
            for (i, j), T_fija in dirichlet.items():
                T[i, j] = T_fija
    p.h_rob = h_orig
    p.dt = dt_orig
    return T

def verif_aislada_sin_fuente(p, k, rhoCp, n_pasos=10):
    """Verif 1: placa aislada, sin fuente, CI uniforme. T constante."""
    S0 = np.zeros((p.Nx, p.Ny))
    T0 = np.full((p.Nx, p.Ny), p.T_inf)
    T = _avanzar_con_S(p, k, rhoCp, T0, S0, n_pasos, h_rob=0.0)
    error = np.max(np.abs(T - T0))
    print("Verif 1 (aislada sin fuente): error max =", error)
    return error

def verif_calentamiento_uniforme(p, k, rhoCp, S0=1.0e5, t_fin=100.0):
    """Verif 2: todo aislado, fuente uniforme S0. T(t) = T0 + S0*t/(rhoCp)."""
    S = np.full((p.Nx, p.Ny), S0)
    T0 = np.full((p.Nx, p.Ny), p.T_inf)
    n_pasos = int(t_fin / p.dt)
    T = _avanzar_con_S(p, k, rhoCp, T0, S, n_pasos, h_rob=0.0)
    T_analitico = T0 + S0 * t_fin / p.rhoCp
    error = np.max(np.abs(T - T_analitico))
    print("Verif 2 (calentamiento uniforme): T_num =", np.max(T),
          "  T_analitico =", np.max(T_analitico),
          "  error =", error)
    return error

def verif_estacionario_dos_materiales(p, k_unused, rhoCp, n_pasos=200):
    """
    Verif 3a: estacionario 1D puro con dos materiales en serie.
    Dirichlet en x=0 (100 C) y x=Lx (0 C), h=0 (bordes y aislados).
    k sintetico: mitades exactas en x.
    Perfil analitico: lineal por tramos con k1*dT1/dx = k2*dT2/dx.
    Usa dt grande para llegar rapido al estacionario.
    """
    # k sintetico: dos mitades exactas
    k = np.full((p.Nx, p.Ny), p.k_base)
    k[p.Nx//2:, :] = p.k_ins

    T_izq = 100.0
    T_der = 0.0
    T = np.full((p.Nx, p.Ny), (T_izq + T_der) / 2.0)

    dirichlet = {}
    for j in range(p.Ny):
        dirichlet[(0, j)] = T_izq
        dirichlet[(p.Nx - 1, j)] = T_der

    S0 = np.zeros((p.Nx, p.Ny))

    # dt grande para alcanzar estacionario rapido
    dt_orig = p.dt
    p.dt = 1000.0
    T = _avanzar_con_S(p, k, rhoCp, T, S0, n_pasos, h_rob=0.0, dirichlet=dirichlet)
    p.dt = dt_orig

    # perfil analitico 1D
    k1 = p.k_base
    k2 = p.k_ins
    q = (T_izq - T_der) / (p.Lx/(2*k1) + p.Lx/(2*k2))
    x_lin = p.x
    T_analitico = np.where(
        x_lin <= p.Lx/2,
        T_izq - q * x_lin / k1,
        T_der + q * (p.Lx - x_lin) / k2
    )
    j = p.Ny // 2
    error = np.max(np.abs(T[:, j] - T_analitico))
    print("Verif 3a (estacionario 2 materiales): error max =", error)
    return error

def verif_balance_energia_estacionario(p, k, rhoCp, n_pasos=200):
    """
    Verif 3b: balance global de energia en estacionario con Robin y fuente.
    En estacionario: P_generada = P_cedida_por_Robin.
    Robin solo en y=0 (cara sur), area A_rob = Lx * e.
    Usa dt grande para llegar rapido al estacionario.
    """
    S0 = 1.0e6   # W/m^3
    S = np.full((p.Nx, p.Ny), S0)
    T0 = np.full((p.Nx, p.Ny), p.T_inf)

    dt_orig = p.dt
    p.dt = 1000.0
    T = _avanzar_con_S(p, k, rhoCp, T0, S, n_pasos, h_rob=p.h_rob)
    p.dt = dt_orig

    # potencia generada total
    P_gen = S0 * p.Lx * p.Ly * p.e

    # potencia cedida por Robin en y=0 (cara sur)
    T_b = T[:, 0]
    P_ced = np.sum(p.h_rob * (T_b - p.T_inf) * (p.dx * p.e))

    error_rel = abs(P_gen - P_ced) / P_gen
    print("Verif 3b (balance estacionario): P_gen =", P_gen,
          "  P_ced =", P_ced, "  error rel =", error_rel)
    return error_rel

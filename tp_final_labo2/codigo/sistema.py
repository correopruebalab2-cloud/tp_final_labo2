import numpy as np
from scipy.sparse import lil_matrix

def fuente_gaussiana(p):
    """s(x,y) normalizada en malla para que sum(s*Vp)=P_total (en pulso)."""
    s = np.exp(-((p.X - p.xs)**2 + (p.Y - p.ys)**2) / (2 * p.sigma**2))
    s *= p.P_total / (s.sum() * p.Vp)
    return s

def f_pulso(t, p):
    return 1.0 if t < p.t_pulso else 0.0

def armar_sistema(p, k, rhoCp, T_old, t, S=None, dirichlet=None):
    """
    Arma A (lil) y b para Euler atras.
    S: fuente volumetrica (Nx,Ny) en W/m^3. Si es None, usa gaussiana * f_pulso.
    dirichlet: dict {(i,j): T_fija}. Si se pasa, esas filas se reemplazan
               por la identidad con b = T_fija.
    """
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    A = lil_matrix((n, n))
    b = np.zeros(n)

    idx = lambda i, j: i * Ny + j

    if S is None:
        s = fuente_gaussiana(p)
        f = f_pulso(t, p)
        S = s * f

    for i in range(Nx):
        for j in range(Ny):
            P = idx(i, j)
            coef_diag = rhoCp[i, j] * p.Vp / p.dt
            b[P] = rhoCp[i, j] * p.Vp / p.dt * T_old[i, j] + S[i, j] * p.Vp

            for (di, dj, A_cara, delta) in [
                ( 1, 0, p.A_cara_x, p.dx),
                (-1, 0, p.A_cara_x, p.dx),
                ( 0, 1, p.A_cara_y, p.dy),
                ( 0,-1, p.A_cara_y, p.dy),
            ]:
                ii, jj = i + di, j + dj
                if 0 <= ii < Nx and 0 <= jj < Ny:
                    G = A_cara / (delta/k[i, j] + delta/k[ii, jj])
                    A[P, idx(ii, jj)] -= G
                    coef_diag += G
                else:
                    if jj == -1 and dj == -1:
                        A_r = p.A_cara_y
                        coef_diag += p.h_rob * A_r
                        b[P] += p.h_rob * A_r * p.T_inf
            A[P, P] = coef_diag

       # --- Imponer Dirichlet en el sistema ---
    if dirichlet is not None:
        for (i, j), T_fija in dirichlet.items():
            P = idx(i, j)
            # borrar la fila P directamente en la estructura lil
            A.rows[P] = [P]
            A.data[P] = [1.0]
            b[P] = T_fija

    return A.tocsr(), b

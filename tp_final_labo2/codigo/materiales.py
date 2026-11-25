import numpy as np
from .malla import mascara_inserto

def campos_materiales(p):
    """k y rhoCp por celda (Nx,Ny)."""
    m = mascara_inserto(p)
    k = np.where(m, p.k_ins, p.k_base)
    rhoCp = np.full_like(k, p.rhoCp)
    return k, rhoCp, m

def conductancia_cara(kP, kE, delta, A):
    """Resistencia en serie: G = A / (delta/kP + delta/kE)."""
    return A / (delta/kP + delta/kE)

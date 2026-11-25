import numpy as np

def mascara_inserto(p):
    """Devuelve máscara booleana (Nx,Ny) True dentro del inserto."""
    a = np.sqrt(p.area_ins * p.ratio_ins)   # lado en x
    b = p.area_ins / a                       # lado en y
    m = (np.abs(p.X - p.xc_ins) <= a/2) & (np.abs(p.Y - p.yc_ins) <= b/2)
    return m

def indices(p):
    """p = i + j*Nx"""
    return np.arange(p.Nx * p.Ny).reshape(p.Nx, p.Ny)

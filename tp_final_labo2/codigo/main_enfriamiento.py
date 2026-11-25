import os
import numpy as np
from .parametros import Parametros
from .materiales import campos_materiales
from .avance import avanzar
from .figuras import (
    mapa_temperatura,
    curva_tmax,
    comparar_tmax,
    comparar_mapas,
)

def asegurar_dirs():
    for d in ["resultados", "figuras"]:
        os.makedirs(d, exist_ok=True)

def correr_caso(p, nombre, con_inserto=True, solver='jacobi_vec'):
    if con_inserto:
        k, rhoCp, m = campos_materiales(p)
    else:
        k = np.full((p.Nx, p.Ny), p.k_base)
        rhoCp = np.full((p.Nx, p.Ny), p.rhoCp)
        m = np.zeros((p.Nx, p.Ny), dtype=bool)
    T0 = np.full((p.Nx, p.Ny), p.T_inf)
    T_hist, tiempos, iters = avanzar(p, k, rhoCp, T0, solver=solver)
    T_max = np.array([np.max(T) for T in T_hist])
    print("Terminado:", nombre, "  Tmax pico =", np.max(T_max), "  Tmax final =", T_max[-1])
    return T_hist, tiempos, iters

def tiempo_enfriamiento(tiempos, T_hist, umbral):
    T_max = np.array([np.max(T) for T in T_hist])
    idx_pico = np.argmax(T_max)
    for k in range(idx_pico, len(T_max)):
        if T_max[k] < umbral:
            return tiempos[k]
    return None

if __name__ == "__main__":
    asegurar_dirs()
    p = Parametros()
    p.t_max = 5000.0
    p.h_rob = 2000.0   # convección fuerte para que cruce el umbral

    disenos = {
        "homogenea":   {"con_inserto": False, "xc": p.xs, "yc": p.ys},
        "ins_centro":  {"con_inserto": True,  "xc": 0.10, "yc": 0.075},
        "ins_entre_robin": {"con_inserto": True, "xc": 0.10, "yc": 0.04},
    }

    resultados = {}
    resumen = []

    for nombre, cfg in disenos.items():
        print("\n=== Corriendo:", nombre, "===")
        p.xc_ins = cfg["xc"]
        p.yc_ins = cfg["yc"]
        T_hist, tiempos, iters = correr_caso(p, nombre, con_inserto=cfg["con_inserto"])
        resultados[nombre] = (tiempos, T_hist)

        carpeta = os.path.join("resultados", nombre + "_h2000")
        os.makedirs(carpeta, exist_ok=True)
        np.save(os.path.join(carpeta, "T_hist.npy"), T_hist)
        np.save(os.path.join(carpeta, "tiempos.npy"), tiempos)

        T_max = np.array([np.max(T) for T in T_hist])
        t_enf = tiempo_enfriamiento(tiempos, T_hist, p.T_inf + p.umbral_enf)
        resumen.append((nombre, np.max(T_max), t_enf))

        n_pasos = len(tiempos) - 1
        idx_pulso = min(int(p.t_pulso / p.dt), n_pasos)
        mapa_temperatura(p, T_hist[idx_pulso], tiempos[idx_pulso], nombre + "_h2000_T_pulso")
        curva_tmax(p, tiempos, T_hist, nombre + "_h2000_Tmax")

    comparar_tmax(p, resultados, "comparacion_Tmax_h2000")
    comparar_mapas(p, resultados, int(p.t_pulso / p.dt), "comparacion_mapas_h2000")

    print("\n=== RESUMEN ===")
    print("Diseno              Tmax_pico(C)   t_enfriamiento(s)")
    for nombre, tmax_pico, t_enf in resumen:
        if t_enf is None:
            t_enf_str = "no alcanza"
        else:
            t_enf_str = str(round(t_enf, 1))
        print(nombre, "   ", round(tmax_pico, 2), "   ", t_enf_str)

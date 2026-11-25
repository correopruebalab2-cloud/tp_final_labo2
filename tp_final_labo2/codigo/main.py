import os
import numpy as np
from .parametros import Parametros
from .materiales import campos_materiales
from .avance import avanzar
from .figuras import (
    mapa_temperatura,
    mapa_flujo,
    curva_tmax,
    comparar_tmax,
    comparar_mapas,
)

def asegurar_dirs():
    for d in ["resultados", "figuras", "informe",
              "datos/originales", "datos/procesados", "documentacion"]:
        os.makedirs(d, exist_ok=True)

def correr_caso(p, nombre, con_inserto=True, solver='jacobi_vec'):
    """Corre un caso. Si con_inserto=False, k uniforme = k_base."""
    if con_inserto:
        k, rhoCp, m = campos_materiales(p)
    else:
        k = np.full((p.Nx, p.Ny), p.k_base)
        rhoCp = np.full((p.Nx, p.Ny), p.rhoCp)
        m = np.zeros((p.Nx, p.Ny), dtype=bool)

    T0 = np.full((p.Nx, p.Ny), p.T_inf)
    T_hist, tiempos, iters = avanzar(p, k, rhoCp, T0, solver=solver)
    T_max = np.array([np.max(T) for T in T_hist])
    print("Terminado:", nombre,
          "  pasos =", len(tiempos),
          "  Tmax pico =", np.max(T_max),
          "  Tmax final =", T_max[-1])
    return T_hist, tiempos, iters, k, rhoCp, m

def tiempo_enfriamiento(tiempos, T_hist, umbral):
    """Devuelve el primer t donde T_max < umbral (despues del pico)."""
    T_max = np.array([np.max(T) for T in T_hist])
    idx_pico = np.argmax(T_max)
    for k in range(idx_pico, len(T_max)):
        if T_max[k] < umbral:
            return tiempos[k]
    return None

if __name__ == "__main__":
    asegurar_dirs()
    p = Parametros()
    p.t_max = 3000.0
    p.h_rob = 200.0          # forzamos por si el parametros.py no se actualizo

    # 6 disenos: homogenea + 5 posiciones del inserto
    disenos = {
        "homogenea":     {"con_inserto": False, "xc": p.xs, "yc": p.ys},
        "ins_centro":    {"con_inserto": True,  "xc": 0.10, "yc": 0.075},
        "ins_despl":     {"con_inserto": True,  "xc": 0.05, "yc": 0.075},
        "ins_esquina":   {"con_inserto": True,  "xc": 0.15, "yc": 0.12},
        "ins_entre_robin": {"con_inserto": True, "xc": 0.10, "yc": 0.04},
        "ins_arriba":    {"con_inserto": True,  "xc": 0.10, "yc": 0.11},
    }

    resultados = {}
    resumen = []

    for nombre, cfg in disenos.items():
        print("\n=== Corriendo:", nombre, "===")
        p.xc_ins = cfg["xc"]
        p.yc_ins = cfg["yc"]
        T_hist, tiempos, iters, k, rhoCp, m = correr_caso(
            p, nombre, con_inserto=cfg["con_inserto"], solver='jacobi_vec'
        )
        resultados[nombre] = (tiempos, T_hist)

        carpeta_caso = os.path.join("resultados", nombre)
        os.makedirs(carpeta_caso, exist_ok=True)
        np.save(os.path.join(carpeta_caso, "T_hist.npy"), T_hist)
        np.save(os.path.join(carpeta_caso, "tiempos.npy"), tiempos)
        np.save(os.path.join(carpeta_caso, "iters.npy"), iters)

        T_max = np.array([np.max(T) for T in T_hist])
        t_pico = tiempos[np.argmax(T_max)]
        t_enf = tiempo_enfriamiento(tiempos, T_hist, p.T_inf + p.umbral_enf)
        resumen.append((nombre, np.max(T_max), t_pico, t_enf))

        n_pasos = len(tiempos) - 1
        idx_pulso = min(int(p.t_pulso / p.dt), n_pasos)
        idx_final = n_pasos
        mapa_temperatura(p, T_hist[idx_pulso], tiempos[idx_pulso],
                         nombre + "_T_pulso")
        mapa_temperatura(p, T_hist[idx_final], tiempos[idx_final],
                         nombre + "_T_final")
        mapa_flujo(p, T_hist[idx_pulso], tiempos[idx_pulso],
                   nombre + "_flujo_pulso")
        curva_tmax(p, tiempos, T_hist, nombre + "_Tmax")

    # figuras comparativas
    print("\n=== Generando figuras comparativas ===")
    comparar_tmax(p, resultados, "comparacion_Tmax")
    idx_pulso_comun = min(int(p.t_pulso / p.dt), len(tiempos) - 1)
    comparar_mapas(p, resultados, idx_pulso_comun, "comparacion_mapas_pulso")

    # tabla resumen
    print("\n=== RESUMEN ===")
    print("Diseno              Tmax_pico(C)   t_pico(s)   t_enfriamiento(s)")
    for nombre, tmax_pico, t_pico, t_enf in resumen:
        if t_enf is None:
            t_enf_str = "no alcanza"
        else:
            t_enf_str = str(round(t_enf, 1))
        print(nombre, "   ", round(tmax_pico, 2), "   ", round(t_pico, 1), "   ", t_enf_str)

    print("\nTodo guardado en resultados/ y figuras/")

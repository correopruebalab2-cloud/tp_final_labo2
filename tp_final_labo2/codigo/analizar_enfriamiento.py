import os
import numpy as np

def analizar_caso(carpeta, nombre):
    tiempos = np.load(os.path.join(carpeta, "tiempos.npy"))
    T_hist = np.load(os.path.join(carpeta, "T_hist.npy"))
    T_max = np.array([np.max(T) for T in T_hist])
    idx_pico = np.argmax(T_max)
    t_pico = tiempos[idx_pico]
    T_pico = T_max[idx_pico]

    print("\n=== ", nombre, " ===")
    print("Tmax pico =", round(T_pico, 2), " C   en t =", round(t_pico, 1), " s")

    for umbral in [40.0, 35.0, 30.0, 25.0, 21.0]:
        t_enf = None
        for k in range(idx_pico, len(T_max)):
            if T_max[k] < umbral:
                t_enf = tiempos[k]
                break
        if t_enf is None:
            print("  umbral", umbral, "C: no alcanza")
        else:
            print("  umbral", umbral, "C: t =", round(t_enf, 1), " s")

if __name__ == "__main__":
    casos = [
        ("resultados/homogenea_h2000", "homogenea h2000"),
        ("resultados/ins_centro_h2000", "ins_centro h2000"),
        ("resultados/ins_entre_robin_h2000", "ins_entre_robin h2000"),
    ]
    for carpeta, nombre in casos:
        if os.path.exists(carpeta):
            analizar_caso(carpeta, nombre)
        else:
            print("No existe:", carpeta)

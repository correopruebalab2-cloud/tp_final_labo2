import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend sin ventana, guarda directo a archivo
import matplotlib.pyplot as plt

def _asegurar_dir(ruta):
    os.makedirs(ruta, exist_ok=True)

def mapa_temperatura(p, T, t, nombre, carpeta="figuras"):
    """
    Mapa de temperatura (pcolormesh) en un instante t.
    """
    _asegurar_dir(carpeta)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(p.x * 1000, p.y * 1000, T.T, shading='auto', cmap='hot')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperatura (°C)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperatura  t = " + str(round(t, 1)) + " s")
    ax.set_aspect('equal')
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

def mapa_flujo(p, T, t, nombre, carpeta="figuras"):
    """
    Mapa de magnitud de flujo |grad T| (K/m) en un instante t.
    """
    _asegurar_dir(carpeta)
    # gradiente con diferencias centradas
    dTdy, dTdx = np.gradient(T, p.dy, p.dx)
    mag = np.sqrt(dTdx**2 + dTdy**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(p.x * 1000, p.y * 1000, mag.T, shading='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|grad T| (K/m)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Flujo de calor  t = " + str(round(t, 1)) + " s")
    ax.set_aspect('equal')
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

def curva_tmax(p, tiempos, T_hist, nombre, carpeta="figuras"):
    """
    Curva T_max(t) vs t. Marca el fin del pulso y el umbral.
    """
    _asegurar_dir(carpeta)
    T_max = np.array([np.max(T) for T in T_hist])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tiempos, T_max, 'b-', label="T_max(t)")
    ax.axvline(p.t_pulso, color='r', linestyle='--', label="fin del pulso")
    ax.axhline(p.T_inf + p.umbral_enf, color='g', linestyle=':', label="umbral")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T_max (°C)")
    ax.set_title("Temperatura maxima en la placa")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

def curva_punto(p, tiempos, T_hist, i, j, nombre, carpeta="figuras"):
    """
    Curva T(t) en un punto (i,j) identificado.
    """
    _asegurar_dir(carpeta)
    T_punto = np.array([T[i, j] for T in T_hist])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tiempos, T_punto, 'b-')
    ax.axvline(p.t_pulso, color='r', linestyle='--', label="fin del pulso")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (°C)")
    ax.set_title("T(t) en (x,y) = (" + str(round(p.x[i]*1000, 1)) + ", " + str(round(p.y[j]*1000, 1)) + ") mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

def comparar_tmax(p, resultados, nombre, carpeta="figuras"):
    """
    Compara T_max(t) de varios disenos.
    resultados: dict {nombre_caso: (tiempos, T_hist)}
    """
    _asegurar_dir(carpeta)
    fig, ax = plt.subplots(figsize=(7, 4))
    for caso, (tiempos, T_hist) in resultados.items():
        T_max = np.array([np.max(T) for T in T_hist])
        ax.plot(tiempos, T_max, label=caso)
    ax.axvline(p.t_pulso, color='k', linestyle='--', alpha=0.5, label="fin del pulso")
    ax.axhline(p.T_inf + p.umbral_enf, color='g', linestyle=':', label="umbral")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T_max (°C)")
    ax.set_title("Comparacion de disenos: T_max(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

def comparar_mapas(p, resultados, t_idx, nombre, carpeta="figuras"):
    """
    Compara mapas de T de varios disenos en el mismo instante (indice t_idx).
    resultados: dict {nombre_caso: (tiempos, T_hist)}
    """
    _asegurar_dir(carpeta)
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (caso, (tiempos, T_hist)) in zip(axes, resultados.items()):
        T = T_hist[t_idx]
        im = ax.pcolormesh(p.x * 1000, p.y * 1000, T.T, shading='auto', cmap='hot',
                           vmin=min(np.min(Th[t_idx]) for _, Th in resultados.values()),
                           vmax=max(np.max(Th[t_idx]) for _, Th in resultados.values()))
        ax.set_title(caso + "\nt = " + str(round(tiempos[t_idx], 1)) + " s")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label="T (°C)")
    fig.tight_layout()
    ruta = os.path.join(carpeta, nombre + ".png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Guardado:", ruta)

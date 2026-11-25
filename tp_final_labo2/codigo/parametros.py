from dataclasses import dataclass, field
import numpy as np

@dataclass
class Parametros:
    # Geometría
    Lx: float = 0.20          # m
    Ly: float = 0.15          # m
    e: float = 0.005          # m (espesor)
    Nx: int = 41              # nodos en x
    Ny: int = 31              # nodos en y

    # Materiales
    k_base: float = 1.0       # W/(m K)
    k_ins: float = 20.0       # W/(m K)
    rhoCp: float = 1.0e6      # J/(m^3 K)

    # Inserto rectangular (área fija)
    area_ins: float = 0.0020  # m^2
    ratio_ins: float = 2.0    # a/b
    xc_ins: float = 0.10      # centro del inserto (se sobreescribe por diseño)
    yc_ins: float = 0.075

    # Fuente gaussiana
    P_total: float = 50.0     # W
    xs: float = 0.10          # m
    ys: float = 0.075         # m
    sigma: float = 0.02       # m
    t_pulso: float = 60.0     # s

    # Robin (borde y=0)
    h_rob: float = 200.0      # antes 50.0      # W/(m^2 K)
    T_inf: float = 20.0       # °C

    # Tiempo
    dt: float = 1.0           # s
    t_max: float = 3000.0     # s (incluye enfriamiento)
    umbral_enf: float = 1.0   # °C sobre T_inf

    # Solver iterativo
    tol: float = 1e-8
    max_iter: int = 10000

    # Derivados
    dx: float = field(init=False)
    dy: float = field(init=False)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)
    Vp: float = field(init=False)
    A_cara_x: float = field(init=False)
    A_cara_y: float = field(init=False)

    def __post_init__(self):
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.Vp = self.dx * self.dy * self.e
        self.A_cara_x = self.dy * self.e   # cara normal a x
        self.A_cara_y = self.dx * self.e   # cara normal a y

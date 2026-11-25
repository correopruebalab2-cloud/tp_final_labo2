from .parametros import Parametros
from .materiales import campos_materiales
from .verificaciones import (
    verif_aislada_sin_fuente,
    verif_calentamiento_uniforme,
    verif_estacionario_dos_materiales,
    verif_balance_energia_estacionario,
)

if __name__ == "__main__":
    p = Parametros()
    k, rhoCp, m = campos_materiales(p)
    e1 = verif_aislada_sin_fuente(p, k, rhoCp)
    e2 = verif_calentamiento_uniforme(p, k, rhoCp)
    e3a = verif_estacionario_dos_materiales(p, k, rhoCp)
    e3b = verif_balance_energia_estacionario(p, k, rhoCp)
    print("Resumen:")
    print("  Verif 1  :", e1)
    print("  Verif 2  :", e2)
    print("  Verif 3a :", e3a)
    print("  Verif 3b :", e3b)

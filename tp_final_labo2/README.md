# TP Final Laboratorio II - Conducción de calor en placa 2D  ( Primera entrega ) 

Trabajo final de la materia Laboratorio II (UNSa). Simulación numérica de la conducción de calor en una placa 2D no homogénea con un inserto conductor, para analizar el efecto de la posición del inserto en la temperatura máxima y el enfriamiento.

## Autor
Renzo Ortiz

## Contenido del repositorio
- `codigo/`: código fuente en Python (módulos de simulación, solvers, figuras).
- `resultados/`: datos numéricos guardados de las corridas (archivos `.npy`).
- `figuras/`: figuras generadas para el informe.
- `informe/`: informe final en PDF y fuente LaTeX.
- `README.md`: este archivo.

## Requisitos
- Python 3.11 o superior
- NumPy
- SciPy
- Matplotlib

Instalación de dependencias:
```bash
pip install numpy scipy matplotlib


## Cómo ejecutar

Desde la raíz del proyecto (`tp_final_labo2/`):

# Correr simulaciones de los 6 diseños
python3 -m codigo.main

# Correr verificaciones del modelo
python3 -m codigo.run_verificaciones

# Generar figuras del caso base
python3 -m codigo.main

# KAN Regression Demo / Demo de Regresion KAN

## ES - Resumen de alto nivel

Este proyecto muestra un ejemplo compacto de regresion con una red Kolmogorov-Arnold (KAN) usando PyTorch.
El script principal genera datos sinteticos, entrena el modelo y reporta metricas de entrenamiento y prueba.

### Objetivo

Aprender una funcion objetivo no lineal de dos variables:

f(x, y) = exp(sin(pi*x) + y^2)

### Que hace el flujo

1. Genera un dataset sintetico de entrenamiento y prueba.
2. Construye un modelo KAN con configuracion simple.
3. Entrena con optimizador LBFGS.
4. Muestra tiempo de entrenamiento, perdida de entrenamiento y perdida de prueba.

### Componentes principales

- kan_modelo.py: Punto de entrada del proyecto.
- model/: Artefactos generados por ejecucion/entrenamiento.

### Requisitos

- Python 3.9+
- torch
- pykan (paquete que expone KAN y create_dataset)

### Ejecucion

```bash
python kan_modelo.py
```

### Salida esperada

En consola se imprime:
- inicio de preparacion de datos
- construccion del modelo
- progreso de entrenamiento
- resumen final con tiempo y perdidas

### Notas de versionado

La carpeta model/ se genera automaticamente y no es necesaria para el codigo fuente.
Por eso se ignora en Git mediante .gitignore.

---

## EN - High-level overview

This project is a compact regression example using a Kolmogorov-Arnold Network (KAN) with PyTorch.
The main script creates synthetic data, trains the model, and prints training/test metrics.

### Goal

Learn a nonlinear target function of two variables:

f(x, y) = exp(sin(pi*x) + y^2)

### Pipeline

1. Generate synthetic train and test data.
2. Build a small KAN model.
3. Train with the LBFGS optimizer.
4. Report training time, train loss, and test loss.

### Main components

- kan_modelo.py: Project entry point.
- model/: Generated runtime/training artifacts.

### Requirements

- Python 3.9+
- torch
- pykan (package exposing KAN and create_dataset)

### Run

```bash
python kan_modelo.py
```

### Expected output

Console output includes:
- data preparation start
- model construction
- training progress
- final summary with runtime and losses

### Version control note

The model/ folder is generated automatically and is not required as source code.
It is ignored in Git through .gitignore.

# -*- coding: utf-8 -*-
"""
config.py
=========
Archivo central de CONFIGURACIÓN del proyecto.

Aquí están todos los hiperparámetros de la GAN condicional (cGAN).
Si quieres experimentar (cambiar la categoría, las épocas, el dataset...),
este es el ÚNICO archivo que necesitas modificar.
"""

import torch

# ----------------------------------------------------------------------
# 1. DATASET Y CATEGORÍA
# ----------------------------------------------------------------------

# Dataset a usar. Opciones: "MNIST" (dígitos) o "FashionMNIST" (ropa).
DATASET = "MNIST"

# Nombres legibles de cada categoría (clase) según el dataset.
# El índice de la lista es el número de la etiqueta (0 a 9).
NOMBRES_CLASES = {
    "MNIST": [f"Digito {i}" for i in range(10)],
    "FashionMNIST": [
        "Camiseta", "Pantalon", "Sueter", "Vestido", "Abrigo",
        "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota",
    ],
}

# Lista de nombres que se usará en todo el proyecto.
CLASES = NOMBRES_CLASES[DATASET]

# Número total de categorías (en MNIST y FashionMNIST son 10).
NUM_CLASES = 10

# CATEGORÍA OBJETIVO: la categoría que se mostrará como resultado principal
# de la GAN. Es un número de 0 a 9 (por ejemplo, 7 = "Digito 7" en MNIST
# o "Zapatilla" en FashionMNIST).
CATEGORIA_OBJETIVO = 7

if not 0 <= CATEGORIA_OBJETIVO < NUM_CLASES:
    raise ValueError("CATEGORIA_OBJETIVO debe estar entre 0 y 9.")

# ----------------------------------------------------------------------
# 2. PARÁMETROS DE LAS REDES NEURONALES
# ----------------------------------------------------------------------

# Tamaño de las imágenes (28 x 28 píxeles, en escala de grises).
IMG_SIZE = 28

# Dimensión del vector de ruido (z) que recibe el generador.
# El generador convierte este vector aleatorio en una imagen.
LATENT_DIM = 100

# Tamaño del vector con el que se representa la etiqueta (categoría).
# La etiqueta (un número) se convierte en un vector con una capa Embedding.
EMBED_DIM = 50

# ----------------------------------------------------------------------
# 3. PARÁMETROS DE ENTRENAMIENTO
# ----------------------------------------------------------------------

# Número de imágenes que se procesan a la vez en cada paso.
BATCH_SIZE = 64

# Número de pasadas completas por todo el dataset.
# Para una prueba rápida usa 3 o 5; para buenos resultados usa 20 o más.
EPOCHS = 20

# Learning rate (tasa de aprendizaje) del optimizador Adam.
LR = 0.0002

# Betas de Adam. Con beta1 = 0.5 el entrenamiento de GANs es más estable.
BETAS = (0.5, 0.999)

# Cada cuántos lotes (batches) se imprime una línea de progreso.
LOG_CADA_N_LOTES = 200

# ----------------------------------------------------------------------
# 4. RESULTADOS Y VISUALIZACIÓN
# ----------------------------------------------------------------------

# Cantidad de imágenes que se muestran por categoría en las figuras.
N_MUESTRAS = 10

# Carpeta donde se guardan las gráficas, imágenes y pesos de las redes.
OUTPUT_DIR = "resultados"

# Carpeta donde se descarga el dataset.
DATA_DIR = "data"

# Si es True, se abren ventanas con las gráficas al final del programa.
# Si es False, solo se guardan como archivos .png en OUTPUT_DIR.
MOSTRAR_GRAFICAS = True

# Semilla para que los resultados sean reproducibles.
SEED = 42

# ----------------------------------------------------------------------
# 5. DISPOSITIVO DE CÓMPUTO
# ----------------------------------------------------------------------

# Usa la GPU (cuda) si está disponible; si no, usa la CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

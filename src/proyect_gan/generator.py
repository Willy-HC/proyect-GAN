# -*- coding: utf-8 -*-
"""
generator.py
============
############################################################
#   AQUÍ SE DEFINE LA RED NEURONAL GENERADORA (Generator)  #
############################################################

¿Qué hace el generador?
-----------------------
Recibe dos cosas:
  1. Un vector de RUIDO aleatorio (z) de tamaño LATENT_DIM.
  2. Una ETIQUETA: la categoría que queremos generar (por ejemplo, el 7).

Y produce una IMAGEN falsa de esa categoría.

Su objetivo es ENGAÑAR al discriminador: que sus imágenes parezcan reales.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Red generadora de la GAN condicional (cGAN)."""

    def __init__(self, latent_dim, num_classes, embed_dim, img_size):
        # Inicializa la clase base de PyTorch (obligatorio en todo nn.Module).
        super().__init__()

        # Guardamos el tamaño de la imagen (28) para reconstruirla al final.
        self.img_size = img_size

        # Número de píxeles de la imagen aplanada: 28 * 28 = 784.
        img_pixels = img_size * img_size

        # --------------------------------------------------------------
        # CAPA DE EMBEDDING PARA LA ETIQUETA
        # --------------------------------------------------------------
        # Convierte la categoría (un entero de 0 a 9) en un vector de
        # tamaño embed_dim. Este vector SE APRENDE durante el entrenamiento.
        # Así el generador "sabe" qué categoría debe dibujar.
        # Parámetros: num_classes * embed_dim = 10 * 50 = 500.
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # --------------------------------------------------------------
        # RED PRINCIPAL (perceptrón multicapa)
        # --------------------------------------------------------------
        # Entrada: ruido (latent_dim = 100) + etiqueta (embed_dim = 50) = 150
        # Salida: imagen aplanada de 784 valores.
        self.model = nn.Sequential(
            # Capa 1: 150 -> 256 neuronas.
            nn.Linear(latent_dim + embed_dim, 256),
            # LeakyReLU: activación que deja pasar una pequeña pendiente
            # negativa (0.2); ayuda a que el entrenamiento de GANs sea estable.
            nn.LeakyReLU(0.2),

            # Capa 2: 256 -> 512 neuronas.
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            # Capa 3: 512 -> 1024 neuronas.
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            # Capa de salida: 1024 -> 784 (un valor por píxel).
            nn.Linear(1024, img_pixels),
            # Tanh deja los píxeles entre -1 y 1, que coincide con la
            # normalización aplicada a las imágenes reales (ver train.py).
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        """
        Paso hacia adelante (forward): genera imágenes.

        noise  : tensor (batch, latent_dim) con ruido aleatorio.
        labels : tensor (batch,) con la categoría de cada imagen a generar.
        return : tensor (batch, 1, 28, 28) con las imágenes generadas.
        """
        # 1) Convertimos cada etiqueta en su vector: (batch, embed_dim).
        label_vectors = self.label_embedding(labels)

        # 2) Unimos ruido y etiqueta en un solo vector: (batch, 150).
        #    Este es el "condicionamiento": el generador recibe la categoría.
        x = torch.cat([noise, label_vectors], dim=1)

        # 3) Pasamos el vector por la red: (batch, 784).
        img = self.model(x)

        # 4) Damos forma de imagen: (batch, 1 canal, 28, 28).
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

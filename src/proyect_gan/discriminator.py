# -*- coding: utf-8 -*-
"""
discriminator.py
================
################################################################
#   AQUÍ SE DEFINE LA RED NEURONAL DISCRIMINADORA (Discriminator) #
################################################################

¿Qué hace el discriminador?
---------------------------
Recibe dos cosas:
  1. Una IMAGEN (real del dataset o falsa creada por el generador).
  2. La ETIQUETA (categoría) que supuestamente tiene esa imagen.

Y devuelve un número entre 0 y 1:
  - Cercano a 1  -> "creo que la imagen es REAL".
  - Cercano a 0  -> "creo que la imagen es FALSA".

Su objetivo es NO dejarse engañar por el generador.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Red discriminadora de la GAN condicional (cGAN)."""

    def __init__(self, num_classes, embed_dim, img_size):
        # Inicializa la clase base de PyTorch (obligatorio en todo nn.Module).
        super().__init__()

        # Guardamos el tamaño de la imagen (28).
        self.img_size = img_size

        # Número de píxeles de la imagen aplanada: 28 * 28 = 784.
        img_pixels = img_size * img_size

        # --------------------------------------------------------------
        # CAPA DE EMBEDDING PARA LA ETIQUETA
        # --------------------------------------------------------------
        # Igual que en el generador: convierte la categoría (0-9) en un
        # vector de tamaño embed_dim. Tiene sus propios pesos, distintos
        # a los del generador. Parámetros: 10 * 50 = 500.
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # --------------------------------------------------------------
        # RED PRINCIPAL (perceptrón multicapa)
        # --------------------------------------------------------------
        # Entrada: imagen aplanada (784) + etiqueta (50) = 834 valores.
        # Salida: 1 valor (probabilidad de que la imagen sea real).
        self.model = nn.Sequential(
            # Capa 1: 834 -> 1024 neuronas.
            nn.Linear(img_pixels + embed_dim, 1024),
            # LeakyReLU: activación estable para GANs.
            nn.LeakyReLU(0.2),
            # Dropout: apaga al azar el 30% de las neuronas en cada paso.
            # Evita que el discriminador "memorice" y se vuelva demasiado
            # fuerte frente al generador.
            nn.Dropout(0.3),

            # Capa 2: 1024 -> 512 neuronas.
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Capa 3: 512 -> 256 neuronas.
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Capa de salida: 256 -> 1 neurona.
            nn.Linear(256, 1),
            # Sigmoid comprime la salida al rango [0, 1] (una probabilidad).
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        """
        Paso hacia adelante (forward): clasifica imágenes como reales/falsas.

        img    : tensor (batch, 1, 28, 28) con las imágenes a evaluar.
        labels : tensor (batch,) con la categoría de cada imagen.
        return : tensor (batch, 1) con la probabilidad de ser real.
        """
        # 1) Aplanamos la imagen: (batch, 1, 28, 28) -> (batch, 784).
        img_flat = img.view(img.size(0), -1)

        # 2) Convertimos cada etiqueta en su vector: (batch, embed_dim).
        label_vectors = self.label_embedding(labels)

        # 3) Unimos imagen y etiqueta: (batch, 834).
        #    Así el discriminador juzga si la imagen es real Y si
        #    corresponde a la categoría indicada.
        x = torch.cat([img_flat, label_vectors], dim=1)

        # 4) Pasamos por la red y obtenemos la probabilidad: (batch, 1).
        validity = self.model(x)
        return validity

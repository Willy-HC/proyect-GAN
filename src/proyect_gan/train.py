# -*- coding: utf-8 -*-
"""
train.py
========
Contiene:
  1. La carga del dataset (MNIST o FashionMNIST).
  2. Las FUNCIONES DE PÉRDIDA de generador y discriminador.
  3. Los optimizadores.
  4. El BUCLE DE ENTRENAMIENTO de la GAN.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if __package__:
    from . import config, visualize
else:
    import config
    import visualize


def cargar_datos():
    """Descarga (si hace falta) y prepara el dataset de imágenes reales."""

    # Transformaciones aplicadas a cada imagen:
    transformacion = transforms.Compose([
        # Convierte la imagen a tensor con valores entre 0 y 1.
        transforms.ToTensor(),
        # Normaliza a valores entre -1 y 1: (x - 0.5) / 0.5.
        # Debe coincidir con la salida Tanh del generador.
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Obtiene la clase del dataset elegido en config.py
    # (datasets.MNIST o datasets.FashionMNIST).
    clase_dataset = getattr(datasets, config.DATASET)

    # Carga el conjunto de entrenamiento (60.000 imágenes).
    dataset = clase_dataset(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transformacion,
    )

    # El DataLoader entrega las imágenes en lotes (batches) mezclados.
    # drop_last=True descarta el último lote si queda incompleto.
    # num_workers=0 evita problemas de multiproceso en Windows.
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    return dataset, dataloader


def entrenar(generador, discriminador, dataloader):
    """
    Entrena la GAN y devuelve un historial con los valores de cada época.

    historial["loss_D"] : pérdida media del discriminador.
    historial["loss_G"] : pérdida media del generador.
    historial["D_x"]    : salida media del discriminador con imágenes REALES.
    historial["D_G_z"]  : salida media del discriminador con imágenes FALSAS.
    """

    # ==================================================================
    # FUNCIÓN DE PÉRDIDA: Binary Cross Entropy (BCE)
    # ==================================================================
    # Mide qué tan lejos está la predicción (0 a 1) de la respuesta
    # correcta (0 = falsa, 1 = real).
    #   BCE(p, y) = -[ y * log(p) + (1 - y) * log(1 - p) ]
    # Si el discriminador acierta, la pérdida es baja; si falla, es alta.
    criterio = nn.BCELoss()

    # ==================================================================
    # OPTIMIZADORES (uno por cada red)
    # ==================================================================
    # Cada red tiene su propio optimizador Adam, que solo actualiza
    # los pesos de SU red.
    optimizador_G = torch.optim.Adam(
        generador.parameters(), lr=config.LR, betas=config.BETAS)
    optimizador_D = torch.optim.Adam(
        discriminador.parameters(), lr=config.LR, betas=config.BETAS)

    # Historial para graficar las pérdidas al final.
    historial = {"loss_D": [], "loss_G": [], "D_x": [], "D_G_z": []}

    # Ruido FIJO: se usa en todas las épocas para generar la categoría
    # objetivo y así ver cómo MEJORAN las imágenes con el entrenamiento.
    ruido_fijo = torch.randn(
        config.N_MUESTRAS, config.LATENT_DIM, device=config.DEVICE)

    n_lotes = len(dataloader)

    # ------------------------------------------------------------------
    # BUCLE PRINCIPAL: una iteración por época
    # ------------------------------------------------------------------
    for epoca in range(1, config.EPOCHS + 1):

        # Acumuladores para calcular el promedio de la época.
        suma_loss_D = suma_loss_G = suma_D_x = suma_D_G_z = 0.0

        for i, (imgs_reales, etiquetas_reales) in enumerate(dataloader, 1):

            # Mueve los datos al dispositivo (CPU o GPU).
            imgs_reales = imgs_reales.to(config.DEVICE)
            etiquetas_reales = etiquetas_reales.to(config.DEVICE)
            batch = imgs_reales.size(0)

            # Respuestas correctas: 1 para "real" y 0 para "falsa".
            objetivo_real = torch.ones(batch, 1, device=config.DEVICE)
            objetivo_falso = torch.zeros(batch, 1, device=config.DEVICE)

            # ==========================================================
            # PASO 1: GENERAR IMÁGENES FALSAS
            # ==========================================================
            # Ruido aleatorio + categorías aleatorias.
            ruido = torch.randn(batch, config.LATENT_DIM, device=config.DEVICE)
            etiquetas_falsas = torch.randint(
                0, config.NUM_CLASES, (batch,), device=config.DEVICE)
            imgs_falsas = generador(ruido, etiquetas_falsas)

            # ==========================================================
            # PASO 2: ENTRENAR EL DISCRIMINADOR
            # ==========================================================
            optimizador_D.zero_grad()  # Borra gradientes anteriores.

            # (a) Con imágenes REALES: debería responder 1.
            salida_real = discriminador(imgs_reales, etiquetas_reales)
            loss_real = criterio(salida_real, objetivo_real)

            # (b) Con imágenes FALSAS: debería responder 0.
            # .detach() corta el gradiente hacia el generador: aquí solo
            # queremos entrenar al discriminador, no al generador.
            salida_falsa = discriminador(imgs_falsas.detach(), etiquetas_falsas)
            loss_falsa = criterio(salida_falsa, objetivo_falso)

            # PÉRDIDA DEL DISCRIMINADOR = pérdida real + pérdida falsa.
            #   loss_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
            loss_D = loss_real + loss_falsa

            loss_D.backward()          # Calcula los gradientes.
            optimizador_D.step()       # Actualiza los pesos de D.

            # ==========================================================
            # PASO 3: ENTRENAR EL GENERADOR
            # ==========================================================
            optimizador_G.zero_grad()  # Borra gradientes anteriores.

            # El generador quiere que el discriminador diga "real" (1)
            # sobre sus imágenes falsas. Ahora SIN detach, para que el
            # error llegue hasta los pesos del generador.
            salida_falsa_G = discriminador(imgs_falsas, etiquetas_falsas)

            # PÉRDIDA DEL GENERADOR:
            #   loss_G = BCE(D(G(z)), 1)
            # Es baja cuando el discriminador se deja engañar.
            loss_G = criterio(salida_falsa_G, objetivo_real)

            loss_G.backward()          # Calcula los gradientes.
            optimizador_G.step()       # Actualiza los pesos de G.

            # ==========================================================
            # REGISTRO DE VALORES
            # ==========================================================
            suma_loss_D += loss_D.item()
            suma_loss_G += loss_G.item()
            suma_D_x += salida_real.mean().item()      # D(x)
            suma_D_G_z += salida_falsa.mean().item()   # D(G(z))

            # Línea de progreso cada cierto número de lotes.
            if i % config.LOG_CADA_N_LOTES == 0:
                print(f"   Epoca {epoca:02d} | lote {i}/{n_lotes} | "
                      f"loss_D: {loss_D.item():.4f} | "
                      f"loss_G: {loss_G.item():.4f}")

        # --------------------------------------------------------------
        # FIN DE LA ÉPOCA: promedios y resumen
        # --------------------------------------------------------------
        historial["loss_D"].append(suma_loss_D / n_lotes)
        historial["loss_G"].append(suma_loss_G / n_lotes)
        historial["D_x"].append(suma_D_x / n_lotes)
        historial["D_G_z"].append(suma_D_G_z / n_lotes)

        print(f"Epoca [{epoca:02d}/{config.EPOCHS}] | "
              f"loss_D: {historial['loss_D'][-1]:.4f} | "
              f"loss_G: {historial['loss_G'][-1]:.4f} | "
              f"D(x): {historial['D_x'][-1]:.4f} | "
              f"D(G(z)): {historial['D_G_z'][-1]:.4f}")

        # Guarda una imagen de la categoría objetivo para ver la evolución.
        visualize.guardar_evolucion(generador, ruido_fijo, epoca)

    return historial

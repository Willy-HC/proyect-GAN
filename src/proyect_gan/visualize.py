# -*- coding: utf-8 -*-
"""
visualize.py
============
Funciones para VER lo que ocurre en el proyecto:
  - Resumen de parámetros de cada red neuronal.
  - Gráficas de las funciones de pérdida y de D(x) / D(G(z)).
  - Imágenes generadas por la GAN (por categoría) frente a imágenes reales.
"""

import os

import matplotlib.pyplot as plt
import torch

if __package__:
    from . import config
else:
    import config


# ----------------------------------------------------------------------
# UTILIDADES INTERNAS
# ----------------------------------------------------------------------

def _a_numpy(imgs):
    """Convierte imágenes de [-1, 1] (salida del generador) a [0, 1]."""
    imgs = (imgs.detach().cpu() + 1) / 2
    return imgs.clamp(0, 1).squeeze(1).numpy()


def _guardar(fig, ruta, mostrar):
    """Guarda la figura como .png y la muestra (o la cierra)."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    if mostrar and config.MOSTRAR_GRAFICAS:
        plt.show()
    plt.close(fig)


def generar_imagenes(generador, categoria, n, ruido=None):
    """Genera n imágenes de UNA categoría usando el generador."""
    generador.eval()                       # Modo evaluación.
    with torch.no_grad():                  # No hace falta calcular gradientes.
        if ruido is None:
            ruido = torch.randn(n, config.LATENT_DIM, device=config.DEVICE)
        etiquetas = torch.full(
            (n,), categoria, dtype=torch.long, device=config.DEVICE)
        imgs = generador(ruido, etiquetas)
    generador.train()                      # Vuelve a modo entrenamiento.
    return imgs


# ----------------------------------------------------------------------
# RESUMEN DE PARÁMETROS DE LAS REDES
# ----------------------------------------------------------------------

def resumen_red(modelo, nombre):
    """Imprime la arquitectura y los parámetros de una red neuronal."""
    print("=" * 70)
    print(f"RED: {nombre}")
    print("=" * 70)
    print(modelo)                                   # Arquitectura completa.
    print("-" * 70)
    print(f"{'Parametro':<35}{'Forma':<20}{'Cantidad':>12}")
    print("-" * 70)
    total = 0
    for nombre_param, p in modelo.named_parameters():
        total += p.numel()
        print(f"{nombre_param:<35}{str(tuple(p.shape)):<20}{p.numel():>12,}")
    print("-" * 70)
    print(f"TOTAL DE PARAMETROS ENTRENABLES: {total:,}")
    print("=" * 70 + "\n")
    return total


# ----------------------------------------------------------------------
# GRÁFICAS DE LAS FUNCIONES DE PÉRDIDA
# ----------------------------------------------------------------------

def graficar_perdidas(historial, ruta):
    """Dibuja la pérdida de G y D, y los valores D(x) y D(G(z))."""
    epocas = range(1, len(historial["loss_D"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Gráfica 1: funciones de pérdida de ambas redes.
    ax1.plot(epocas, historial["loss_D"], label="Perdida Discriminador (loss_D)")
    ax1.plot(epocas, historial["loss_G"], label="Perdida Generador (loss_G)")
    ax1.set_title("Funciones de perdida")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Perdida (BCE)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Gráfica 2: qué tan "real" cree el discriminador cada tipo de imagen.
    ax2.plot(epocas, historial["D_x"], label="D(x): imagenes REALES")
    ax2.plot(epocas, historial["D_G_z"], label="D(G(z)): imagenes FALSAS")
    ax2.axhline(0.5, color="gray", linestyle="--", label="0.5 (equilibrio)")
    ax2.set_title("Salida del discriminador")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Probabilidad de ser real")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend()

    _guardar(fig, ruta, mostrar=True)


# ----------------------------------------------------------------------
# IMÁGENES GENERADAS
# ----------------------------------------------------------------------

def guardar_evolucion(generador, ruido_fijo, epoca):
    """Guarda las imágenes de la categoría objetivo en una época dada."""
    imgs = _a_numpy(generar_imagenes(
        generador, config.CATEGORIA_OBJETIVO, config.N_MUESTRAS, ruido_fijo))

    fig, axes = plt.subplots(1, config.N_MUESTRAS, figsize=(config.N_MUESTRAS, 1.5))
    for ax, img in zip(axes, imgs):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Epoca {epoca} - Categoria: "
                 f"{config.CLASES[config.CATEGORIA_OBJETIVO]}", fontsize=10)

    ruta = os.path.join(config.OUTPUT_DIR, "evolucion", f"epoca_{epoca:02d}.png")
    _guardar(fig, ruta, mostrar=False)


def mostrar_resultado_categoria(generador, dataset, ruta):
    """Compara imágenes GENERADAS vs REALES de la categoría objetivo."""
    categoria = config.CATEGORIA_OBJETIVO
    n = config.N_MUESTRAS

    # Imágenes generadas por la GAN.
    falsas = _a_numpy(generar_imagenes(generador, categoria, n))

    # Imágenes reales de esa misma categoría (tomadas del dataset).
    reales = dataset.data[dataset.targets == categoria][:n].numpy() / 255.0

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 3.2))
    for j in range(n):
        axes[0, j].imshow(falsas[j], cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(reales[j], cmap="gray")
        axes[1, j].axis("off")
    axes[0, 0].set_title("GENERADAS (GAN)", loc="left", fontsize=10)
    axes[1, 0].set_title("REALES (dataset)", loc="left", fontsize=10)
    fig.suptitle(f"Resultado de la GAN - Categoria usada: "
                 f"{config.CLASES[categoria]} (etiqueta {categoria})",
                 fontsize=12)

    _guardar(fig, ruta, mostrar=True)


def mostrar_todas_las_categorias(generador, ruta):
    """Muestra una fila de imágenes generadas por cada categoría."""
    n = config.N_MUESTRAS
    fig, axes = plt.subplots(config.NUM_CLASES, n, figsize=(n * 1.1, config.NUM_CLASES * 1.1))

    for categoria in range(config.NUM_CLASES):
        imgs = _a_numpy(generar_imagenes(generador, categoria, n))
        for j in range(n):
            ax = axes[categoria, j]
            ax.imshow(imgs[j], cmap="gray")
            ax.axis("off")
        # Nombre de la categoría a la izquierda de cada fila.
        axes[categoria, 0].text(
            -0.15, 0.5, config.CLASES[categoria], transform=axes[categoria, 0].transAxes,
            ha="right", va="center", fontsize=9)

    fig.suptitle("Imagenes generadas por la GAN en todas las categorias", fontsize=12)
    _guardar(fig, ruta, mostrar=True)

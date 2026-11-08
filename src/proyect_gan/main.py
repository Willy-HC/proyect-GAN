# -*- coding: utf-8 -*-
"""
main.py
=======
PUNTO DE ENTRADA DEL PROYECTO. Ejecuta todo en este orden:

  1. Carga el dataset.
  2. CREA la red generadora y la red discriminadora.
  3. Muestra los parámetros de cada red.
  4. Muestra los valores iniciales (antes de entrenar).
  5. Entrena la GAN (funciones de pérdida en train.py).
  6. Muestra las gráficas y el resultado de la GAN por categoría.

Ejecución:  python main.py
"""

import os

import torch

if __package__:
    from . import config, train, visualize
    from .discriminator import Discriminator
    from .generator import Generator
else:
    import config
    import train
    import visualize
    from discriminator import Discriminator
    from generator import Generator


def mostrar_valores_iniciales(generador, discriminador, dataloader):
    """Muestra qué valores producen las redes ANTES de entrenar."""
    print("=" * 70)
    print("VALORES INICIALES (redes sin entrenar)")
    print("=" * 70)

    # Tomamos un lote de imágenes reales del dataset.
    imgs_reales, etiquetas = next(iter(dataloader))
    imgs_reales = imgs_reales.to(config.DEVICE)
    etiquetas = etiquetas.to(config.DEVICE)

    # Modo evaluación (desactiva Dropout) y sin calcular gradientes.
    generador.eval()
    discriminador.eval()
    with torch.no_grad():
        ruido = torch.randn(len(imgs_reales), config.LATENT_DIM, device=config.DEVICE)
        imgs_falsas = generador(ruido, etiquetas)

        d_real = discriminador(imgs_reales, etiquetas).mean().item()
        d_falsa = discriminador(imgs_falsas, etiquetas).mean().item()
    generador.train()
    discriminador.train()

    print(f"Entrada del generador   : ruido {tuple(ruido.shape)} + etiquetas {tuple(etiquetas.shape)}")
    print(f"Salida del generador    : imagenes {tuple(imgs_falsas.shape)} "
          f"(min {imgs_falsas.min():.3f}, max {imgs_falsas.max():.3f}, media {imgs_falsas.mean():.3f})")
    print(f"Discriminador D(x)      : {d_real:.4f}  (promedio sobre imagenes REALES)")
    print(f"Discriminador D(G(z))   : {d_falsa:.4f}  (promedio sobre imagenes FALSAS)")
    print("Sin entrenar, ambos valores estan cerca de 0.5: el discriminador aun no distingue.")
    print("=" * 70 + "\n")


def main():
    # Semilla para que los resultados sean reproducibles.
    torch.manual_seed(config.SEED)

    # Carpeta donde se guardan los resultados.
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Dispositivo      : {config.DEVICE}")
    print(f"Dataset          : {config.DATASET}")
    print(f"Categoria usada  : {config.CLASES[config.CATEGORIA_OBJETIVO]} "
          f"(etiqueta {config.CATEGORIA_OBJETIVO})")
    print(f"Epocas           : {config.EPOCHS}\n")

    # ------------------------------------------------------------------
    # PASO 1: CARGAR EL DATASET
    # ------------------------------------------------------------------
    dataset, dataloader = train.cargar_datos()

    # ------------------------------------------------------------------
    # PASO 2: CREACIÓN DE LAS REDES NEURONALES
    # ------------------------------------------------------------------

    # ##################################################################
    # #   AQUÍ SE CREA LA RED NEURONAL GENERADORA                      #
    # #   (la clase está definida en generator.py)                     #
    # ##################################################################
    generador = Generator(
        latent_dim=config.LATENT_DIM,    # Tamaño del ruido de entrada.
        num_classes=config.NUM_CLASES,   # Número de categorías.
        embed_dim=config.EMBED_DIM,      # Tamaño del vector de la etiqueta.
        img_size=config.IMG_SIZE,        # Tamaño de la imagen (28).
    ).to(config.DEVICE)                  # Envía la red a CPU o GPU.

    # ##################################################################
    # #   AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA                  #
    # #   (la clase está definida en discriminator.py)                 #
    # ##################################################################
    discriminador = Discriminator(
        num_classes=config.NUM_CLASES,   # Número de categorías.
        embed_dim=config.EMBED_DIM,      # Tamaño del vector de la etiqueta.
        img_size=config.IMG_SIZE,        # Tamaño de la imagen (28).
    ).to(config.DEVICE)                  # Envía la red a CPU o GPU.

    # ------------------------------------------------------------------
    # PASO 3: PARÁMETROS DE CADA RED
    # ------------------------------------------------------------------
    visualize.resumen_red(generador, "GENERADOR")
    visualize.resumen_red(discriminador, "DISCRIMINADOR")

    # ------------------------------------------------------------------
    # PASO 4: VALORES INICIALES (ANTES DE ENTRENAR)
    # ------------------------------------------------------------------
    mostrar_valores_iniciales(generador, discriminador, dataloader)

    # ------------------------------------------------------------------
    # PASO 5: ENTRENAMIENTO (las funciones de pérdida están en train.py)
    # ------------------------------------------------------------------
    print("INICIANDO ENTRENAMIENTO...\n")
    historial = train.entrenar(generador, discriminador, dataloader)
    print("\nENTRENAMIENTO TERMINADO.\n")

    # ------------------------------------------------------------------
    # PASO 6: RESULTADOS
    # ------------------------------------------------------------------
    # Guarda los pesos entrenados de ambas redes.
    torch.save(generador.state_dict(), os.path.join(config.OUTPUT_DIR, "generador.pth"))
    torch.save(discriminador.state_dict(), os.path.join(config.OUTPUT_DIR, "discriminador.pth"))

    # Gráfica de las funciones de pérdida y de D(x) / D(G(z)).
    visualize.graficar_perdidas(
        historial, os.path.join(config.OUTPUT_DIR, "perdidas.png"))

    # Resultado principal: imágenes generadas vs reales de la categoría usada.
    visualize.mostrar_resultado_categoria(
        generador, dataset, os.path.join(config.OUTPUT_DIR, "resultado_categoria.png"))

    # Resultado general: todas las categorías.
    visualize.mostrar_todas_las_categorias(
        generador, os.path.join(config.OUTPUT_DIR, "resultado_todas_categorias.png"))

    print(f"Listo. Los resultados estan en la carpeta '{config.OUTPUT_DIR}/'.")


# Este bloque es necesario en Windows y evita ejecuciones accidentales
# si otro archivo importa main.py.
if __name__ == "__main__":
    main()

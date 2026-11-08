# GAN Condicional (cGAN) en PyTorch: proyecto educativo

Proyecto en Python que explica cómo funciona una **Red Generativa Antagónica (GAN)**: cómo se crean la **red generadora** y la **red discriminadora**, cuáles son sus **parámetros**, cómo trabajan las **funciones de pérdida** y cuál es el **resultado** para una **categoría específica**.

Todo el código está comentado en español, línea por línea.

---

## 1. ¿Qué es una GAN?

Una GAN está formada por **dos redes neuronales que compiten entre sí**:

| Red | Rol | Analogía |
|---|---|---|
| **Generador (G)** | Crea imágenes falsas a partir de ruido aleatorio | Un falsificador de billetes |
| **Discriminador (D)** | Decide si una imagen es real o falsa | La policía que detecta billetes falsos |

- El **generador** intenta que sus imágenes parezcan reales.
- El **discriminador** intenta no dejarse engañar.
- Al competir, **ambos mejoran**: con el tiempo el generador produce imágenes cada vez más realistas.

### ¿Por qué "condicional"?

Una GAN normal genera imágenes al azar. Esta es una **GAN condicional (cGAN)**: además del ruido, el generador recibe una **etiqueta** (la categoría), así que podemos **pedirle qué categoría generar** (por ejemplo, "dibuja un 7" o "dibuja una zapatilla"). El discriminador también recibe la etiqueta, para comprobar que la imagen sea real **y** corresponda a esa categoría.

---

## 2. Estructura del proyecto

```
proyect-GAN/
├── .python-version
├── .gitignore
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── README.md
└── src/
    └── proyect_gan/
        ├── __init__.py
        ├── config.py          # Hiperparámetros y categoría usada
        ├── generator.py       # AQUÍ se define la red generadora
        ├── discriminator.py   # AQUÍ se define la red discriminadora
        ├── train.py           # Pérdidas y bucle de entrenamiento
        ├── visualize.py       # Resumen, gráficas e imágenes
        └── main.py            # Punto de entrada: crea las redes y ejecuta todo
```

### ¿Dónde se crea cada red neuronal?

| Red | Se **define** (clase) en | Se **crea** (instancia) en |
|---|---|---|
| Generador | `src/proyect_gan/generator.py` → `class Generator` | `src/proyect_gan/main.py` → bloque *"AQUÍ SE CREA LA RED NEURONAL GENERADORA"* |
| Discriminador | `src/proyect_gan/discriminator.py` → `class Discriminator` | `src/proyect_gan/main.py` → bloque *"AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA"* |

Ambos puntos están marcados con comentarios destacados en el código.

---

## 3. Las redes neuronales y sus parámetros

### Generador

Recibe **ruido (100 valores) + etiqueta (embedding de 50 valores)** = 150 valores y produce una imagen de 28×28.

```
Embedding(10 -> 50)
Linear(150 -> 256)  + LeakyReLU
Linear(256 -> 512)  + LeakyReLU
Linear(512 -> 1024) + LeakyReLU
Linear(1024 -> 784) + Tanh   -> imagen 28x28
```

- **Tanh** final: deja los píxeles entre -1 y 1 (igual que las imágenes reales normalizadas).
- Tiene aproximadamente **1,5 millones de parámetros**.

### Discriminador

Recibe **imagen aplanada (784) + etiqueta (embedding de 50)** = 834 valores y devuelve **un número entre 0 y 1**.

```
Embedding(10 -> 50)
Linear(834 -> 1024) + LeakyReLU + Dropout(0.3)
Linear(1024 -> 512) + LeakyReLU + Dropout(0.3)
Linear(512 -> 256)  + LeakyReLU + Dropout(0.3)
Linear(256 -> 1)    + Sigmoid   -> probabilidad de ser real
```

- **Sigmoid** final: convierte la salida en una probabilidad (1 = real, 0 = falsa).
- **Dropout**: evita que el discriminador se vuelva demasiado fuerte.
- Tiene aproximadamente **1,5 millones de parámetros**.

Al ejecutar el programa, se imprime la **arquitectura completa y la cantidad exacta de parámetros por capa** de ambas redes.

---

## 4. Funciones de pérdida

Se usa **Binary Cross Entropy (BCE)**, que mide qué tan lejos está la predicción (entre 0 y 1) de la respuesta correcta (0 o 1):

```
BCE(p, y) = -[ y·log(p) + (1 - y)·log(1 - p) ]
```

Cada red tiene su propia pérdida:

### Pérdida del discriminador

```
loss_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
```

- `D(x)`: la salida del discriminador para una imagen **real** → debería ser **1**.
- `D(G(z))`: la salida del discriminador para una imagen **falsa** → debería ser **0**.

### Pérdida del generador

```
loss_G = BCE(D(G(z)), 1)
```

El generador **quiere que el discriminador diga "real"** (1) sobre sus imágenes falsas. Si lo consigue, `loss_G` baja.

### Ciclo de cada paso de entrenamiento

1. El generador crea imágenes falsas con ruido y categorías aleatorias.
2. Se entrena el **discriminador** con imágenes reales y falsas (`loss_D`).
3. Se entrena el **generador** para engañar al discriminador (`loss_G`).

---

## 5. Valores que se muestran

### Antes de entrenar
- Forma de la entrada y salida del generador.
- Rango de los píxeles generados.
- `D(x)` y `D(G(z))`, ambos cerca de 0,5 (el discriminador aún no distingue nada).

### Durante el entrenamiento (por época)

```
Epoca [05/20] | loss_D: 1.0842 | loss_G: 1.2310 | D(x): 0.6521 | D(G(z)): 0.3410
```

| Valor | Significado |
|---|---|
| `loss_D` | Pérdida media del discriminador |
| `loss_G` | Pérdida media del generador |
| `D(x)` | Qué tan "real" cree el discriminador que son las imágenes **reales** |
| `D(G(z))` | Qué tan "real" cree el discriminador que son las imágenes **falsas** |

> Los números del ejemplo son solo ilustrativos; los tuyos variarán.

### Cómo interpretar los valores

- Una GAN sana suele tener `D(x)` algo mayor que 0,5 y `D(G(z))` algo menor que 0,5, acercándose entre sí con el tiempo.
- Si `loss_D` cae casi a 0 y `D(G(z))` casi a 0: el **discriminador gana demasiado** y el generador no aprende.
- Si `D(G(z))` sube a 1 muy rápido: el **generador engaña** al discriminador con imágenes poco realistas (posible *mode collapse*).
- En las GAN **las pérdidas no tienen que bajar siempre**: oscilan porque cada red mejora a costa de la otra. Lo importante es que las imágenes mejoren.

---

## 6. Resultados que genera el programa

Todo se guarda en la carpeta `resultados/`:

| Archivo | Contenido |
|---|---|
| `perdidas.png` | Gráfica de `loss_D`, `loss_G`, `D(x)` y `D(G(z))` por época |
| `resultado_categoria.png` | **Resultado principal**: imágenes generadas vs. reales de la **categoría usada** |
| `resultado_todas_categorias.png` | Una fila de imágenes generadas por cada categoría |
| `evolucion/epoca_XX.png` | Cómo mejoran las imágenes de la categoría usada época tras época |
| `generador.pth` | Pesos entrenados del generador |
| `discriminador.pth` | Pesos entrenados del discriminador |

La **categoría usada** se indica en el título de las figuras y en la consola.

---

## 7. Requisitos

- **Python 3.12 o superior**
- **Visual Studio Code** con la extensión *Python* (de Microsoft)
- Conexión a internet la primera vez (para descargar el dataset y las librerías)
- Una GPU es opcional: el proyecto funciona en CPU

---

## 8. Instalación y ejecución paso a paso

### Paso 1: Abrir el proyecto
Abre la raíz del proyecto `proyect-GAN` en VS Code (*Archivo → Abrir carpeta*).

### Paso 2: Crear un entorno virtual
En la terminal de VS Code (*Terminal → Nueva terminal*):

```bash
python -m venv .venv
```

### Paso 3: Activar el entorno virtual

- **Windows (PowerShell o CMD):**
  ```bash
  .venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```

Verás `(.venv)` al inicio de la línea de la terminal.

> Si PowerShell bloquea la activación, ejecuta esto una vez y vuelve a intentarlo:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Paso 4: Instalar las librerías

```bash
pip install -e .
```

Las dependencias están declaradas en `pyproject.toml`. Si tienes GPU NVIDIA y quieres usarla, consulta https://pytorch.org para instalar la versión de PyTorch adecuada para tu equipo.

### Paso 5: Elegir el intérprete en VS Code
Pulsa `Ctrl+Shift+P` → **Python: Select Interpreter** → elige el intérprete de la carpeta `.venv`.

### Paso 6: Ejecutar

```bash
proyect-gan
```

También puedes usar `python -m proyect_gan.main`. Para ejecutarlo directamente desde el código fuente, desde la raíz del proyecto ejecuta `python src\proyect_gan\main.py` (en macOS/Linux, `python src/proyect_gan/main.py`). Ejecuta siempre desde la raíz para que el dataset y los resultados se guarden en `data/` y `resultados/` dentro del proyecto.

La primera vez se descargará el dataset en la carpeta `data/`.

### Paso 7: Ver los resultados
- En la consola verás los parámetros de las redes, los valores iniciales y las pérdidas por época.
- Al final se abrirán ventanas con las gráficas (ciérralas para que el programa termine).
- Todo queda guardado en `resultados/`.

---

## 9. Cómo personalizar el proyecto (`src/proyect_gan/config.py`)

| Parámetro | Qué hace | Valor por defecto |
|---|---|---|
| `DATASET` | `"MNIST"` (dígitos) o `"FashionMNIST"` (ropa) | `"MNIST"` |
| `CATEGORIA_OBJETIVO` | Categoría que se muestra como resultado (0 a 9) | `7` |
| `EPOCHS` | Número de épocas de entrenamiento | `20` |
| `BATCH_SIZE` | Imágenes por lote | `64` |
| `LR` | Learning rate | `0.0002` |
| `LATENT_DIM` | Tamaño del vector de ruido | `100` |
| `EMBED_DIM` | Tamaño del vector de la etiqueta | `50` |
| `MOSTRAR_GRAFICAS` | `False` para solo guardar las imágenes sin abrir ventanas | `True` |

### Categorías disponibles

**MNIST:** 0 a 9 → Dígito 0, 1, 2, … 9.

**FashionMNIST:**

| Etiqueta | Categoría | Etiqueta | Categoría |
|---|---|---|---|
| 0 | Camiseta | 5 | Sandalia |
| 1 | Pantalón | 6 | Camisa |
| 2 | Suéter | 7 | Zapatilla |
| 3 | Vestido | 8 | Bolso |
| 4 | Abrigo | 9 | Bota |

### Ejemplo: generar zapatillas
En `src/proyect_gan/config.py` cambia:

```python
DATASET = "FashionMNIST"
CATEGORIA_OBJETIVO = 7
```

---

## 10. Consejos

- **Prueba rápida:** pon `EPOCHS = 3` para comprobar que todo funciona. En CPU cada época puede tardar de unas decenas de segundos a un par de minutos, según tu equipo.
- **Mejores resultados:** usa `EPOCHS = 30` o más.
- Las imágenes de las primeras épocas son manchas; mejoran progresivamente. Revisa la carpeta `resultados/evolucion/` para verlo.

---

## 11. Problemas frecuentes

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | El entorno virtual no está activado o no se instalaron las librerías (pasos 3 y 4). |
| Error al descargar el dataset | Revisa tu conexión a internet y vuelve a ejecutar. |
| El programa parece "colgado" al final | Hay ventanas de gráficas abiertas; ciérralas, o pon `MOSTRAR_GRAFICAS = False`. |
| Las imágenes salen como manchas | Entrena más épocas. Las GAN necesitan tiempo. |
| El entrenamiento es muy lento | Reduce `EPOCHS`, o usa una GPU o Google Colab. |

---

## 12. Conceptos clave resumidos

- **GAN:** dos redes que compiten; una genera, la otra discrimina.
- **cGAN:** la GAN recibe la categoría, así que podemos pedir qué generar.
- **Embedding:** convierte la etiqueta (un número) en un vector que la red aprende.
- **BCE:** función de pérdida que mide el error de una clasificación real/falsa.
- **`D(x)` y `D(G(z))`:** qué tan real cree el discriminador que son las imágenes reales y las falsas.
- **`.detach()`:** evita que el entrenamiento del discriminador modifique al generador.

# Predicción del consumo de frío — Planta cervecera

Trabajo Práctico Final de Laboratorio de Datos II. A partir de los totalizadores horarios de una planta cervecera (Tecate, México) se predice el **consumo de frío (kW) del día siguiente**.

Los comandos están para **Windows** (CMD / PowerShell / Anaconda Prompt) y **Linux** (bash). Cuando difieren, se muestran las dos versiones. Todos se corren desde la **carpeta raíz del repo**.

---

## Requisitos

- [Git](https://git-scm.com/downloads)
- **Python 3.12**, con una de estas dos opciones:
  - [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html) (recomendado, mismos comandos en Windows y Linux), o
  - Python 3.12 instalado en el sistema (se usa `venv`).
- Los **3 Excels de totalizadores** (no están en el repo).

---

## 1. Abrir una terminal

- **Windows:** si usás conda, abrir **Anaconda Prompt** (buscar "Anaconda Prompt" en el menú de inicio). Si no, CMD o PowerShell.
- **Linux:** cualquier terminal.

## 2. Clonar el repositorio

```bash
git clone https://github.com/Marcjun/Cerveceria-LabDatosII
cd Cerveceria-LabDatosII
```

## 3. Crear el entorno e instalar dependencias

### Opción A — Conda (Windows y Linux)

```bash
conda create -n cervecera_env python=3.12 -y
conda activate cervecera_env
pip install uv
uv pip install -r requirements.txt
```

Cada vez que se abra una terminal nueva hay que volver a activarlo:

```bash
conda activate cervecera_env
```

> Con el entorno activado, `python` apunta al Python del entorno tanto en Windows como en Linux, así que los comandos siguientes son iguales en ambos.

## 4. Agregar los datos y verificar los checksums

Copiar los 3 Excels en `data/raw/` con **estos nombres exactos** (un guion bajo entre los años, sin espacios alrededor):

```
data/raw/Totalizadores Planta 2020_2022.xlsx
data/raw/Totalizadores Planta 2021_2023.xlsx
data/raw/Totalizadores Planta 2022_2023.xlsx
```

- Si los archivos vienen con otro nombre (por ejemplo `2020 _ 2022` o `2020-2022`), renombrarlos: los scripts buscan `Totalizadores Planta AAAA_AAAA.xlsx`.
- **En Linux importan las mayúsculas:** tiene que ser `Totalizadores Planta`, con `T` y `P` mayúsculas, y la extensión `.xlsx` en minúscula.
- En Windows, cerrar los Excels antes de correr los scripts.

Verificar que son los mismos archivos que se usaron en el TP:

```bash
python src/auxiliar_functions.py verify
```

Los 3 `.xlsx` tienen que coincidir. Los archivos de `data/processed/` que todavía no se generaron aparecen como `FALTA`: es normal en un clon nuevo. Si algún archivo sale como `DISTINTO`, el contenido no es el esperado y el comando termina con error.

## 5. Correr el pipeline de preprocesamiento

```bash
python src/preprocessing_pipeline.py
```

Lee los Excels de `data/raw/`, aplica las decisiones guardadas en `data/processed/tratamientos_EDA.json` y `data/processed/tratamientos_PREPROCESSING.json`, y genera:

| Archivo | Qué es |
|---|---|
| `data/processed/datos_preprocess.csv` | Dataset diario con la limpieza del EDA (entrada de la notebook de preprocesamiento) |
| `data/processed/dataset_final.csv` | Dataset final con features y target |
| `data/processed/data_lineage.json` | Registro de cada transformación aplicada en la corrida |

Volver a verificar: ahora `datos_preprocess.csv` y `dataset_final.csv` tienen que coincidir. `data_lineage.json` va a salir como `DISTINTO` (y el comando termina con error) porque guarda la fecha y hora de la corrida; eso es esperable.

```bash
python src/auxiliar_functions.py verify
```

## 6. Predecir

```bash
python src/predict.py "data/raw/Totalizadores Planta 2022_2023.xlsx"
```

- Recibe la ruta de **un** Excel con la misma estructura que los de entrenamiento (puede ser uno nuevo, en cualquier carpeta). Las **comillas** son necesarias porque el nombre tiene espacios, en Windows y en Linux.
- Opcionalmente se puede indicar la versión del modelo (por defecto, la última de `models/model_registry.json`):
  ```bash
  python src/predict.py "data/raw/Totalizadores Planta 2022_2023.xlsx" 4.2.0
  ```
- Escribe `results/predicciones.csv`: una fila por día, `fecha` = día D y `prediccion_frio_kw` = consumo de frío estimado para D+1.

No hace falta correr el pipeline del paso 5 antes de predecir: `predict.py` aplica el mismo preprocesamiento al Excel que recibe.

## 7. Notebooks (opcional)

La EDA lee los CSV formateados de cada Excel, que se generan con:

```bash
python src/procesado_raw.py
```

Abrir Jupyter desde la raíz del repo (con el entorno activado):

```bash
jupyter lab
```

En VS Code, abrir la carpeta del repo y elegir como kernel el entorno `cervecera_env` (o `.venv`).

Orden: `notebooks/eda.ipynb` → `notebooks/preprocesado.ipynb` → `notebooks/modelado.ipynb`.

> **Atención:** correr las notebooks reescribe archivos versionados (`tratamientos_EDA.json`, `tratamientos_PREPROCESSING.json`, `results/experiment_logs.csv`). Para solo reproducir los resultados alcanzan los pasos 4 a 6. Después de cambiar algo en las notebooks hay que volver a correr el pipeline (paso 5) y actualizar los checksums con `python src/auxiliar_functions.py update`.

## 8. Reentrenar el modelo final (opcional)

```
python src/train_model.py <version> <modelo> <hiperparametros en JSON>
```

Modelos disponibles: `ridge`, `lasso`, `random_forest`, `xgboost`, `lightgbm`. Mide en validación (entrenado solo con train), reentrena con train + validación, guarda `models/modelo_v<version>.pkl` (borra los `.pkl` anteriores) y actualiza `models/model_registry.json`. Necesita `data/processed/dataset_final.csv` (paso 5).

El JSON de hiperparámetros se escribe distinto según la terminal:

**Linux (bash) y PowerShell 7.3 o posterior:**
```bash
python src/train_model.py 3.0.1 ridge '{"alpha": 10}'
```

**Windows CMD / Anaconda Prompt:**
```bat
python src/train_model.py 3.0.1 ridge "{\"alpha\": 10}"
```

**Windows PowerShell 5 (el que viene con Windows):**
```powershell
python src/train_model.py 3.0.1 ridge '{\"alpha\": 10}'
```

---

## Estructura del repo

```
Cerveceria-LabDatosII/
├── data/
│   ├── checksums.json                  # SHA-256 de los datos (verify / update)
│   ├── raw/                            # Excels originales (no se versionan)
│   ├── external/temperatura_tecate.csv # temperatura diaria (Open-Meteo)
│   └── processed/                      # JSON de decisiones + salidas del pipeline
├── notebooks/
│   ├── eda.ipynb                       # escribe tratamientos_EDA.json
│   ├── preprocesado.ipynb              # escribe tratamientos_PREPROCESSING.json
│   └── modelado.ipynb                  # baselines, Optuna, elección del modelo
├── src/
│   ├── procesado_raw.py                # Excel -> CSV formateado (para la EDA)
│   ├── preprocessing_pipeline.py       # Excels -> dataset_final.csv + data_lineage.json
│   ├── aux_functions_preprocess.py     # funciones del pipeline
│   ├── auxiliar_functions.py           # checksums
│   ├── train_model.py                  # entrenamiento y registro del modelo
│   └── predict.py                      # predicción del día siguiente
├── models/                             # modelo entrenado + model_registry.json
├── results/                            # experiment_logs.csv, predicciones.csv
└── requirements.txt
```

---

## Problemas comunes

| Problema | Solución |
|---|---|
| `conda` no se reconoce (Windows) | Usar **Anaconda Prompt**, o correr una vez `conda init cmd.exe` / `conda init powershell` y reabrir la terminal |
| `py` no se reconoce (Windows) | Usar `python -m venv .venv` (verificando antes con `python --version` que sea 3.12) |
| `python` no existe (Linux) | Activar el entorno primero; fuera de él usar `python3` |
| Error `libgomp.so.1` al importar LightGBM (Linux) | `sudo apt install libgomp1` |
| El pipeline no encuentra los Excels | Revisar nombres y mayúsculas en `data/raw/` (paso 4) |
| `PermissionError` al leer un Excel (Windows) | Cerrar el archivo en Excel |
| `verify` marca `DISTINTO` en un `.xlsx` | El archivo no es el original del TP: volver a descargarlo |

"""Predicción del consumo de frío del día siguiente

    python src/predict.py <ruta al Excel> [version]

Lee un Excel con la misma estructura que los de entrenamiento, le aplica el mismo
preprocesamiento (reusa preprocessing_pipeline.py y los dos JSON de tratamientos) y
escribe results/predicciones.csv: una fila por día, `fecha` = día D y
`prediccion_frio_kw` = consumo de frío estimado para D+1.
"""
from pathlib import Path
import json
import sys
import warnings

import joblib
import pandas as pd

from aux_functions_preprocess import ErrorEstructura, leer_tratamiento, leer_temperatura, columnas_requeridas
from preprocessing_pipeline import construir_dataset, preprocesar


def raiz_repo():
    """Carpeta raíz del repo (src/..), para que las rutas no dependan de dónde se corre."""
    return Path(__file__).resolve().parent.parent


def load_model(version="latest"):
    """Carga el modelo (Pipeline completo: escalador si lo lleva + estimador) desde el registro.
    'latest' carga la versión semántica más alta (clave "latest" de models/model_registry.json).
    Devuelve (modelo, info): info es la entrada del registro (features, hiperparámetros, path...)."""
    ruta_registro = raiz_repo() / "models" / "model_registry.json"
    if not ruta_registro.is_file():
        raise FileNotFoundError(f"No existe el registro de modelos: {ruta_registro}. "
                                "Se genera con src/train_model.py.")
    registro = json.loads(ruta_registro.read_text(encoding="utf-8"))

    if version == "latest":
        version = registro["latest"]
    if version not in registro["versiones"]:
        raise ValueError(f"La versión '{version}' no está en el registro. "
                         f"Versiones disponibles: {sorted(registro['versiones'])}")
    info = registro["versiones"][version]

    ruta_modelo = raiz_repo() / info["path"]
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"El registro apunta a {ruta_modelo} (versión {version}), pero el archivo no existe.")
    modelo = joblib.load(ruta_modelo)
    return modelo, {"version": version, **info}


def load_and_preprocess(filepath):
    """Carga el Excel y aplica todo el preprocesamiento reusando preprocessing_pipeline.py:
    limpieza del EDA (tratamientos_EDA.json), outliers, features, calendario y selección
    (tratamientos_PREPROCESSING.json). No ajusta nada con los datos que recibe.
    Devuelve la matriz de características y las fechas (día D) de cada fila."""
    proc = raiz_repo() / "data" / "processed"
    registro_eda = leer_tratamiento(proc / "tratamientos_EDA.json")
    registro = leer_tratamiento(proc / "tratamientos_PREPROCESSING.json")
    temp = leer_temperatura(raiz_repo() / registro["features"]["temperatura"]["csv"])


    # Hojas y columnas que tiene que traer el Excel: las de origen de las features seleccionadas
    requeridas = columnas_requeridas(registro["seleccion"]["features"], registro)
    # con_target=False: no hay día siguiente y no se pierde el último día
    diario = construir_dataset(filepath, registro_eda, con_target=False, requeridas=requeridas)
    final = preprocesar(diario, registro, temp)

    fechas = final["DIA"].reset_index(drop=True)
    X = final.drop(columns=["DIA"]).reset_index(drop=True)
    return X, fechas


def predict_consumption(filepath, version="latest"):
    """Función principal de predicción. Una fila por día:
    'fecha' es el día D y la predicción corresponde al consumo de frío del día D+1."""
    modelo, info = load_model(version)
    X, fechas = load_and_preprocess(filepath)

    # Las features del modelo tienen que coincidir con las que deja el preprocesamiento actual
    # (si se volvió a correr la notebook de preprocesamiento después de entrenar, pueden no coincidir)
    faltan = [c for c in info["features"] if c not in X.columns]
    if faltan:
        raise ErrorEstructura(
            f"El modelo v{info['version']} usa {len(faltan)} features que el preprocesamiento actual no genera "
            f"(¿cambió tratamientos_PREPROCESSING.json después de entrenar?):\n"
            + "\n".join(f"  - {c}" for c in faltan))
    X = X[info["features"]]          # mismo orden que en el entrenamiento

    con_nulos = X.columns[X.isna().any()]
    if len(con_nulos):
        warnings.warn(f"{X.isna().any(axis=1).sum()} días tienen nulos en {len(con_nulos)} features "
                      f"({', '.join(con_nulos)}). Se predicen igual; revisar esas columnas en el Excel.")

    predicciones = modelo.predict(X)
    return pd.DataFrame({"fecha": fechas, "prediccion_frio_kw": predicciones})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python src/predict.py <ruta al Excel> [version]   (version por defecto: latest)")
    version = sys.argv[2] if len(sys.argv) > 2 else "latest"

    try:
        results = predict_consumption(sys.argv[1], version)
    except (FileNotFoundError, ValueError) as e:      # ErrorEstructura es un ValueError
        sys.exit(f"No se pudo predecir. {type(e).__name__}: {e}")
    except KeyError as e:
        sys.exit(f"No se pudo predecir: falta la columna o clave {e}. "
                 "Revisar que el Excel tenga la misma estructura que los de entrenamiento.")

    salida = raiz_repo() / "results" / "predicciones.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(salida, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    print(f"{len(results)} predicciones ({results['fecha'].min().date()} -> {results['fecha'].max().date()}) "
          f"generadas en {salida}")
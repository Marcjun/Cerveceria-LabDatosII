"""Entrenamiento de modelos

Funciones que usa notebooks/modelado.ipynb (baselines, búsqueda con Optuna, métricas y
registro de experimentos) y entrenamiento final del modelo elegido:

    python src/train_model.py <version> <modelo> <hiperparametros en JSON>

mide en validación con el modelo entrenado solo con train, reentrena con train + validación, guarda
models/modelo_v<version>.pkl y actualiza models/model_registry.json. El test no se usa acá.
"""
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from aux_functions_preprocess import leer_tratamiento, separar_conjuntos


def raiz_repo():
    """Carpeta raíz del repo (src/..), para que las rutas no dependan de dónde se corre."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- datos

def cargar_conjuntos(col_fecha="DIA", col_target="target"):
    """Lee dataset_final.csv y lo corta en train / val / test con las fechas del split
    guardadas en tratamientos_PREPROCESSING.json (no recalcula el corte).
    Devuelve (conjuntos, features): dict de DataFrames completos y la lista de features
    seleccionadas, en el orden del JSON."""
    proc = raiz_repo() / "data" / "processed"
    ruta = proc / "dataset_final.csv"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}. Se genera corriendo src/preprocessing_pipeline.py.")
    datos = pd.read_csv(ruta, parse_dates=[col_fecha])
    registro = leer_tratamiento(proc / "tratamientos_PREPROCESSING.json")

    features = registro["seleccion"]["features"]
    faltan = [c for c in features + [col_target] if c not in datos.columns]
    if faltan:
        raise KeyError(f"dataset_final.csv no tiene {len(faltan)} columnas que pide el JSON: {faltan}. "
                       "Volver a correr el pipeline.")

    conjuntos = separar_conjuntos(datos.sort_values(col_fecha), registro, col_fecha)
    for nombre, df in conjuntos.items():
        print(f"{nombre}: {len(df)} días | {df[col_fecha].min().date()} -> {df[col_fecha].max().date()}")
    return conjuntos, features


# ---------------------------------------------------------------- baselines

def baseline_persistencia(datos, frio="Consolidado EE | Frio (Kw)"):
    """ŷ[D+1] = Frio[D]."""
    return datos[frio].rename("persistencia")


def baseline_media_movil(datos, previo=None, dias=7, frio="Consolidado EE | Frio (Kw)"):
    """ŷ[D+1] = promedio de Frio de D-6 a D (los 7 días anteriores a D+1).
    previo: días anteriores al conjunto, para que los primeros días tengan su ventana completa
    (validación recibe train; test recibe train + validación)."""
    serie = datos[frio] if previo is None else pd.concat([previo[frio], datos[frio]])
    media = serie.rolling(dias, min_periods=1).mean().iloc[-len(datos):]
    return pd.Series(media.to_numpy(), index=datos.index, name=f"media_movil_{dias}")


# ---------------------------------------------------------------- modelos y métricas

def evaluar(y_real, y_pred):
    """MAE, RMSE y R²."""
    return {"mae": float(mean_absolute_error(y_real, y_pred)),
            "rmse": float(root_mean_squared_error(y_real, y_pred)),
            "r2": float(r2_score(y_real, y_pred))}


def armar_modelo(nombre, params, semilla=42):
    """Pipeline de scikit-learn de cada modelo. params son los hiperparámetros elegidos
    (los de Optuna); acá se agregan los fijos (semilla, hilos, silencio).
    Los lineales llevan StandardScaler; los de árboles no lo necesitan."""
    if nombre == "ridge":
        return Pipeline([("escalador", StandardScaler()),
                         ("modelo", Ridge(**params, random_state=semilla))])
    if nombre == "lasso":
        return Pipeline([("escalador", StandardScaler()),
                         ("modelo", Lasso(**{"max_iter": 10000, **params}, random_state=semilla))])
    if nombre == "random_forest":
        return Pipeline([("modelo", RandomForestRegressor(**params, random_state=semilla, n_jobs=-1))])
    if nombre == "xgboost":
        return Pipeline([("modelo", XGBRegressor(**params, random_state=semilla, n_jobs=-1))])
    if nombre == "lightgbm":
        # subsample solo tiene efecto con subsample_freq > 0.
        # deterministic + force_row_wise: dos corridas seguidas dan lo mismo aun con varios hilos.
        return Pipeline([("modelo", LGBMRegressor(**{"subsample_freq": 1, **params}, random_state=semilla,
                                                   n_jobs=-1, deterministic=True, force_row_wise=True,
                                                   verbose=-1))])
    raise ValueError(f"Modelo desconocido: {nombre!r}. Opciones: ridge, lasso, random_forest, xgboost, lightgbm.")


def entrenar_y_evaluar(nombre, params, ajuste, evaluacion, features, conjunto, semilla=42,
                       col_target="target", registrar=True):
    """Entrena con 'ajuste', mide en 'evaluacion' y, si registrar, lo agrega al log
    con el nombre de 'conjunto' ("validacion" / "test"). Devuelve (modelo, predicciones, métricas)."""
    modelo = armar_modelo(nombre, params, semilla)
    modelo.fit(ajuste[features], ajuste[col_target])
    pred = pd.Series(modelo.predict(evaluacion[features]), index=evaluacion.index, name=nombre)
    metricas = evaluar(evaluacion[col_target], pred)
    if registrar:
        registrar_experimento(nombre, conjunto, metricas, hiperparametros=params,
                              n_features=len(features), semilla=semilla)
    return modelo, pred, metricas


# ---------------------------------------------------------------- búsqueda con Optuna

def optimizar(nombre, espacio, X, y, n_trials=50, n_folds=4, dias_fold=91, semilla=42, registrar=True, timeout=None):
    """Busca hiperparámetros con Optuna sobre TimeSeriesSplit de train (validación no se toca).

    espacio: función trial -> dict con los hiperparámetros a probar (se define en la notebook).
    Cada trial entrena fold por fold; después de cada fold informa el MAE promedio acumulado
    y MedianPruner corta los trials que vienen peor que la mediana.
    Cada trial completo queda en results/experiment_logs.csv con conjunto = "cv_train".
    Devuelve el study (study.best_params, study.best_value, study.trials_dataframe())."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(TimeSeriesSplit(n_splits=n_folds, test_size=dias_fold).split(X))
    commit = commit_actual()

    def objetivo(trial):
        params = espacio(trial)
        maes, rmses, r2s = [], [], []
        for paso, (idx_ajuste, idx_eval) in enumerate(folds):
            modelo = armar_modelo(nombre, params, semilla)
            modelo.fit(X.iloc[idx_ajuste], y.iloc[idx_ajuste])
            m = evaluar(y.iloc[idx_eval], modelo.predict(X.iloc[idx_eval]))
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            r2s.append(m["r2"])
            trial.report(float(np.mean(maes)), paso)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("mae_desvio_folds", float(np.std(maes)))
        trial.set_user_attr("rmse", float(np.mean(rmses)))
        trial.set_user_attr("r2", float(np.mean(r2s)))
        return float(np.mean(maes))

    def registrar_trial(study, trial):
        if registrar and trial.state == optuna.trial.TrialState.COMPLETE:
            registrar_experimento(nombre, "cv_train", {"mae": trial.value, **trial.user_attrs},
                                  hiperparametros=trial.params, n_features=X.shape[1],
                                  trial=trial.number, semilla=semilla, commit=commit)

    study = optuna.create_study(direction="minimize", study_name=nombre,
                                sampler=optuna.samplers.TPESampler(seed=semilla),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objetivo, n_trials=n_trials, timeout=timeout, callbacks=[registrar_trial], show_progress_bar=True)

    estados = pd.Series([t.state.name for t in study.trials]).value_counts()
    print(f"{nombre}: {estados.get('COMPLETE', 0)} completos, {estados.get('PRUNED', 0)} podados | "
          f"MAE CV {study.best_value:,.0f} ± {study.best_trial.user_attrs['mae_desvio_folds']:,.0f} kW")
    return study


# ---------------------------------------------------------------- tracking y registro

def commit_actual():
    """Hash corto del commit actual y si hay cambios sin commitear en archivos versionados."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=raiz_repo(), capture_output=True,
                                  text=True, check=True).stdout.strip()
        return {"hash": git("rev-parse", "--short", "HEAD"),
                "sin_commitear": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"hash": "desconocido", "sin_commitear": None}


def registrar_experimento(modelo, conjunto, metricas, hiperparametros=None, n_features=None,
                          trial=None, semilla=42, commit=None):
    """Agrega una fila a results/experiment_logs.csv (lo crea con encabezado si no existe).
    En 'commit', un * al final indica cambios sin commitear al momento de la corrida."""
    ruta = raiz_repo() / "results" / "experiment_logs.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    commit = commit or commit_actual()
    fila = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modelo": modelo,
        "conjunto": conjunto,
        "trial": trial,
        "hiperparametros": json.dumps(hiperparametros or {}, ensure_ascii=False),
        "mae": metricas.get("mae"),
        "mae_desvio_folds": metricas.get("mae_desvio_folds"),
        "rmse": metricas.get("rmse"),
        "r2": metricas.get("r2"),
        "n_features": n_features,
        "semilla": semilla,
        "commit": commit["hash"] + ("*" if commit["sin_commitear"] else ""),
    }
    pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not ruta.exists(), index=False)


def guardar_modelo(modelo, version):
    """Guarda el Pipeline entrenado en models/modelo_v<version>.pkl con joblib."""
    ruta = raiz_repo() / "models" / f"modelo_v{version}.pkl"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    return ruta


def actualizar_registro(version, ruta_modelo, nombre, hiperparametros, features, metricas_val,
                        entrenamiento, semilla=42, col_fecha="DIA"):
    """Agrega (o reemplaza) la versión en models/model_registry.json y actualiza "latest"
    a la versión semántica más alta. Formato: {"latest": "x.y.z", "versiones": {"x.y.z": {...}}}."""
    raiz = raiz_repo()
    ruta = raiz / "models" / "model_registry.json"
    texto = ruta.read_text(encoding="utf-8").strip() if ruta.is_file() else ""
    versiones = json.loads(texto).get("versiones", {}) if texto else {}
    if version in versiones:
        print(f"Aviso: la versión {version} ya estaba en el registro y se reemplaza.")

    commit = commit_actual()
    if commit["sin_commitear"]:
        print("Aviso: hay cambios sin commitear; el hash del registro no reproduce exactamente este modelo.")
    versiones[version] = {
        "path": ruta_modelo.relative_to(raiz).as_posix(),
        "modelo": nombre,
        "hiperparametros": hiperparametros,
        "features": features,
        "metricas_validacion": metricas_val,
        "entrenado_con": {"conjuntos": "train + validacion",
                          "desde": str(entrenamiento[col_fecha].min().date()),
                          "hasta": str(entrenamiento[col_fecha].max().date()),
                          "dias": len(entrenamiento)},
        "semilla": semilla,
        "commit": commit["hash"],
        "cambios_sin_commitear": commit["sin_commitear"],
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    latest = max(versiones, key=lambda v: tuple(int(p) for p in v.split(".")))
    registro = {"latest": latest, "versiones": versiones}
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


# ---------------------------------------------------------------- entrenamiento final

if __name__ == "__main__":
    semilla = 42

    if len(sys.argv) != 4:
        raise SystemExit("Uso: python src/train_model.py <version> <modelo> <hiperparametros en JSON>")
    version, nombre, params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

    print(f"Modelo elegido: {nombre} | {params}")

    conjuntos, features = cargar_conjuntos()

    # Métricas de validación con el modelo entrenado solo con train (las que van al registro)
    _, _, metricas_val = entrenar_y_evaluar(nombre, params, conjuntos["train"], conjuntos["val"],
                                            features, "validacion", semilla, registrar=False)
    print(f"Validación: MAE {metricas_val['mae']:,.0f} | RMSE {metricas_val['rmse']:,.0f} | "
          f"R² {metricas_val['r2']:.3f}")

    # Modelo final: train + validación (punto 3.4)
    entrenamiento = pd.concat([conjuntos["train"], conjuntos["val"]], ignore_index=True)
    modelo = armar_modelo(nombre, params, semilla)
    modelo.fit(entrenamiento[features], entrenamiento["target"])

    ruta_modelo = guardar_modelo(modelo, version)
    ruta_registro = actualizar_registro(version, ruta_modelo, nombre, params, features,
                                        metricas_val, entrenamiento, semilla)
    print(f"Guardados {ruta_modelo.name} y {ruta_registro.name} (versión {version})")
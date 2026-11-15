from pathlib import Path
import warnings

import pandas as pd

from datetime import datetime
import json
from pathlib import Path
import warnings

import pandas as pd

from aux_functions_preprocess import (
    HOJAS_EXCLUIDAS, HOJAS_REQUERIDAS, ErrorEstructura,
    # Unificación de los Excels
    procesar_hoja, unir_hojas, validar_estructura,
    corregir_dias_pegados, recortar_julio_agosto, sacar_dias_cortos,
    filtrar_2359, convertir_a_float, unir_archivos,
    rellenar_dias_faltantes, agregar_target, agregar_calendario,
    # Limpieza
    leer_tratamiento, borrar_columnas, marcar_sin_meta,
    columnas_con_tratamiento, marcar_nulos_por_area, imputar_promedio_anteriores,
    # Split y outliers
    separar_conjuntos, corregir_outliers,
    # Feature engineering
    leer_temperatura, agregar_temperatura, media_movil, variacion, rezago,
    agregar_combinaciones, agregar_eficiencias,
    # Selección y chequeo
    seleccionar, chequear_conjuntos,
)


def leer_excel(ruta):
    """Lee todas las hojas y las devuelve por nombre (sin espacios de más),
    sin las de HOJAS_EXCLUIDAS (Auxiliar y Seguimiento Dia, que el EDA descarta)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    hojas = pd.read_excel(ruta, sheet_name=None, engine="calamine")
    hojas = {str(nombre).strip(): df for nombre, df in hojas.items()}
    return {nombre: df for nombre, df in hojas.items() if nombre not in HOJAS_EXCLUIDAS}


def formatear_excel(ruta):
    """Excel crudo -> un df con DIA, HORA y "<Hoja> | <columna>"."""
    hojas = leer_excel(ruta)
    validar_estructura(hojas)

    limpias = {}
    for nombre, df in hojas.items():
        try:
            df_limpio, eliminadas = procesar_hoja(df, nombre)
        except ErrorEstructura as e:
            if nombre in HOJAS_REQUERIDAS:
                raise
            warnings.warn(f"{e} -> se saltea la hoja.")
            continue
        if df_limpio.empty and nombre in HOJAS_REQUERIDAS:
            raise ErrorEstructura(f"Hoja '{nombre}': no tiene ninguna fila con DIA y HORA válidos.")
        limpias[nombre] = df_limpio
    unido = unir_hojas(limpias)

    # Filas sin ningún dato aparte de DIA y HORA. En el EDA eran las 20 filas que
    # solo tenían Auxiliar (2020_2022); como Auxiliar ya no se lee, no deberían
    # aparecer, pero se borran igual por si alguna hoja trae filas vacías.
    datos = unido.columns.difference(["DIA", "HORA"])
    return unido.dropna(how="all", subset=datos).reset_index(drop=True)


def cargar_excels(rutas):
    """Formatea cada Excel por separado. Devuelve {nombre_archivo: df} en orden
    cronológico (por nombre de archivo), como el dict `dfs` del EDA.
    Acepta una ruta sola (predict.py) o varias (entrenamiento)."""
    if isinstance(rutas, (str, Path)):
        rutas = [rutas]
    rutas = sorted(Path(r) for r in rutas)
    if not rutas:
        raise FileNotFoundError("No se encontró ningún Excel para procesar.")
    return {ruta.stem: formatear_excel(ruta) for ruta in rutas}

def preparar_archivos(dfs):
    """Pasos del EDA que se hacen archivo por archivo, en el mismo orden que la notebook."""
    preparados = {}
    for i, (nombre, df) in enumerate(dfs.items()):
        df = corregir_dias_pegados(df)
        if i > 0:                       # con un solo Excel (predict.py) no se aplica
            df = recortar_julio_agosto(df)
        df = sacar_dias_cortos(df)
        df = filtrar_2359(df)
        if df.empty:
            raise ErrorEstructura(
                f"'{nombre}': después de la limpieza no quedó ningún día con lectura de cierre "
                "(HORA entre 23:59:00 y 23:59:59). Sin esa lectura no se puede armar el total diario.")
        preparados[nombre] = convertir_a_float(df)
    return preparados


def armar_dataset(dfs, con_target=True):
    """dict `dfs` (salida de cargar_excels) -> una fila por día, como datos_base.csv del EDA.
    con_target=False para predict.py: no hay día siguiente y no se puede perder el último día."""
    df = rellenar_dias_faltantes(unir_archivos(preparar_archivos(dfs)))
    df = agregar_calendario(df)
    return agregar_target(df) if con_target else df


def construir_dataset(rutas, con_target=True):
    """Excel(s) crudo(s) -> una fila por día. Punto de entrada para train y predict."""
    return armar_dataset(cargar_excels(rutas), con_target)


def guardar_dataset(df, ruta):
    """Guarda el dataset diario como CSV (DIA en formato aaaa-mm-dd, sin índice),
    con el mismo formato que datos_base.csv del EDA. Devuelve la ruta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, date_format="%Y-%m-%d")
    return ruta


def limpiar(diario, registro):
    """Limpieza de la notebook de preprocesamiento, leyendo las decisiones del JSON."""
    df = borrar_columnas(diario, registro, "nulos_mayor_36")
    df = marcar_sin_meta(df)
    df = borrar_columnas(df, registro, "nulos_uno_o_dos_tramos")

    cols = columnas_con_tratamiento(df, registro, "promedio_anteriores")
    df = marcar_nulos_por_area(df, cols)          # antes de imputar
    df = imputar_promedio_anteriores(df, cols)

    cols = columnas_con_tratamiento(df, registro, "promedio_anteriores_sin_negativos")
    df = imputar_promedio_anteriores(df, cols, invalidar_negativos=True)

    for paso in ("restos_redundantes", "columnas_en_cero", "columnas_identicas"):
        df = borrar_columnas(df, registro, paso)
    return df


def tratar_outliers(datos, registro):
    """Las dos pasadas de z-score de la notebook, con los números de train del registro."""
    df = corregir_outliers(datos, registro, "pasada1")
    return corregir_outliers(df, registro, "pasada2")


def crear_features(datos, registro, temp):
    """Feature engineering de la notebook, en el orden de la clave "features" del registro.
    Una feature se saltea si falta su columna de origen (el Excel de test puede traer menos columnas);
    si era una de las seleccionadas, `seleccionar` avisa cuál falta. Al final borra las metas."""
    f = registro["features"]
    df = agregar_temperatura(datos, temp)

    for nombre, (columna, ventana) in f["medias_moviles"].items():
        if columna in df.columns:
            df = media_movil(df, columna, nombre, ventana)
    for nombre, columna in f["variaciones"].items():
        if columna in df.columns:
            df = variacion(df, columna, nombre)
    for nombre, (columna, k) in f["rezagos"].items():
        if columna in df.columns:
            df = rezago(df, columna, nombre, k)

    ratios = {nombre: par for nombre, par in f["ratios"].items() if set(par) <= set(df.columns)}
    df = agregar_combinaciones(df, ratios=ratios)
    df = agregar_eficiencias(df, f["eficiencias"], registro["medias_train"]["eficiencias"])

    return borrar_columnas(df, registro, "metas_tras_eficiencias")


def preprocesar(diario, registro, temp):
    """Dataset diario (salida de construir_dataset) -> DIA + features seleccionadas (+ target).
    Aplica lo que decidió la notebook leyendo el registro: no ajusta nada con los datos que recibe."""
    df = limpiar(diario, registro)
    df = tratar_outliers(df, registro)
    df = crear_features(df, registro, temp)
    return seleccionar(df, registro)


def armar_ejecucion(final, conjuntos, entradas):
    """Datos de la corrida del pipeline, para la clave "ejecucion" de data_lineage.json.
    Las transformaciones (qué se borró, imputó, corrigió y creó) están en el resto del archivo."""
    return {
        "script": "src/preprocessing_pipeline.py",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "entradas": sorted(entradas),
        "salida": {"archivo": "data/processed/dataset_final.csv", "filas": len(final),
                   "columnas": final.shape[1], "escalado": False},
        "conjuntos": {nombre: {"X_shape": [len(df), df.shape[1] - 2],   # sin DIA ni target
                               "desde": str(df["DIA"].min().date()),
                               "hasta": str(df["DIA"].max().date())}
                      for nombre, df in conjuntos.items()},
    }


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    data_dir = raiz / "data"
    rutas = list((data_dir / "raw").glob("Totalizadores Planta *.xlsx"))

    dfs = cargar_excels(rutas)
    for nombre, df in dfs.items():
        print(f"{nombre}: {df.shape[0]} filas x {df.shape[1]} cols")

    dataset = armar_dataset(dfs)
    print(f"Dataset diario: {dataset.shape} | {dataset['DIA'].min().date()} -> {dataset['DIA'].max().date()}")

    # Entrada de la notebook de preprocesamiento: se guarda ANTES de limpiar,
    # para que la notebook siga recibiendo el diario sin tocar (1213 x 397)
    salida = guardar_dataset(dataset, data_dir / "processed" / "datos_preprocess.csv")
    print("Guardado en", salida)

    ruta_registro = data_dir / "processed" / "data_lineage.json"
    registro = leer_tratamiento(ruta_registro)
    temp = leer_temperatura(raiz / registro["features"]["temperatura"]["csv"])
    final = preprocesar(dataset, registro, temp)

    conjuntos = separar_conjuntos(final, registro)
    chequear_conjuntos(conjuntos)
    for nombre, df in conjuntos.items():
        X = df.drop(columns=["DIA", "target"])
        print(f"X_{nombre}: {X.shape} | {df['DIA'].min().date()} -> {df['DIA'].max().date()}")

    salida = guardar_dataset(final, data_dir / "processed" / "dataset_final.csv")
    print("Guardado en", salida)

    # Solo se agrega/actualiza "ejecucion": las decisiones de la notebook quedan como están
    registro["ejecucion"] = armar_ejecucion(final, conjuntos, [r.name for r in rutas])
    ruta_registro.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Ejecución registrada en", ruta_registro)
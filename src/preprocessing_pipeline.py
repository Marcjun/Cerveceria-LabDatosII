from datetime import datetime
from pathlib import Path
import warnings
import json
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
    seleccionar, chequear_conjuntos, aplicar, registrar_paso
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

def preparar_archivos(dfs, log=None):
    """Pasos del EDA que se hacen archivo por archivo, en el mismo orden que la notebook."""
    preparados = {}
    for i, (nombre, df) in enumerate(dfs.items()):
        etapa = f"Unión | {nombre}"
        df = aplicar(log, df, corregir_dias_pegados, etapa=etapa,
                     descripcion="Días con más de 26 lecturas: separa los días pegados, saca los bloques copiados y recoloca las fechas",
                     parametros={"max_lecturas": 26})
        if i > 0:                       # con un solo Excel (predict.py) no se aplica
            df = aplicar(log, df, recortar_julio_agosto, etapa=etapa,
                         descripcion="Saca julio y agosto del primer año del archivo (se toman del archivo anterior)")
        df = aplicar(log, df, sacar_dias_cortos, etapa=etapa,
                     descripcion="Saca los días con menos de 20 lecturas", parametros={"min_lecturas": 20})
        df = aplicar(log, df, filtrar_2359, etapa=etapa,
                     descripcion="Deja solo la lectura de cierre del día", parametros={"horas": "23:59:00 a 23:59:59"})
        if df.empty:
            raise ErrorEstructura(
                f"'{nombre}': después de la limpieza no quedó ningún día con lectura de cierre "
                "(HORA entre 23:59:00 y 23:59:59). Sin esa lectura no se puede armar el total diario.")
        preparados[nombre] = aplicar(log, df, convertir_a_float, etapa=etapa,
                                     descripcion="Pasa a float todas las columnas menos DIA y HORA")
    return preparados


def unir(preparados, log=None):
    """Une los archivos en una fila por día y registra el paso."""
    unido = unir_archivos(preparados)
    registrar_paso(log, pd.concat(preparados.values(), ignore_index=True), unido, "Unión",
                   "Une los archivos, saca HORA y las columnas Unnamed y deja una fila por día: "
                   "gana la fila con menos nulos (empate: la primera) y sus huecos se rellenan con las otras",
                   funcion="unir_archivos")
    return unido


def limpiar(diario, registro_eda, log=None):
    """Limpieza del EDA, leyendo sus decisiones de tratamientos_EDA.json."""
    etapa, fuente = "Limpieza (EDA)", "tratamientos_EDA.json"
    df = aplicar(log, diario, marcar_sin_meta, etapa=etapa,
                 descripcion="Agrega no_hay_meta (día sin ninguna meta) y rellena las metas nulas con 1 (provisorio)")
    df = aplicar(log, df, marcar_nulos_por_area, registro_eda["indicadores_nulos"], etapa=etapa,   # antes de imputar
                 descripcion="Agrega nulo_<área>: 1 si ese día faltaba algún dato del área",
                 detalle=f"{fuente} > indicadores_nulos")

    cols = columnas_con_tratamiento(df, registro_eda, "promedio_anteriores")
    df = aplicar(log, df, imputar_promedio_anteriores, cols, etapa=etapa,
                 descripcion="Rellena los nulos con el promedio de los 2 valores válidos anteriores",
                 parametros={"n": 2, "columnas": len(cols)}, detalle=f"{fuente} > tratamientos")
    cols = columnas_con_tratamiento(df, registro_eda, "promedio_anteriores_sin_negativos")
    df = aplicar(log, df, imputar_promedio_anteriores, cols, invalidar_negativos=True, etapa=etapa,
                 descripcion="Pasa los negativos a nulo y los rellena con el promedio de los 2 valores válidos anteriores",
                 parametros={"n": 2, "columnas": len(cols)}, detalle=f"{fuente} > tratamientos")

    # Todas las listas del JSON del EDA son pasos de borrado (incluye correlacion_mayor_<umbral>)
    for paso, columnas in registro_eda.items():
        if isinstance(columnas, list):
            df = aplicar(log, df, borrar_columnas, registro_eda, paso, etapa=etapa,
                         descripcion=f"Borra las columnas del paso '{paso}'",
                         parametros={"columnas_registradas": len(columnas)}, detalle=f"{fuente} > {paso}")
    return df


def armar_dataset(dfs, registro_eda, con_target=True, log=None):
    """dict `dfs` (salida de cargar_excels) -> una fila por día con la limpieza del EDA,
    calendario y target. Es lo que se guarda en datos_preprocess.csv.
    con_target=False para predict.py: no hay día siguiente y no se puede perder el último día."""
    df = unir(preparar_archivos(dfs, log), log)
    df = limpiar(df, registro_eda, log)
    k = registro_eda["dias_faltantes"]["k"]
    df = aplicar(log, df, rellenar_dias_faltantes, k, etapa="Limpieza (EDA)",
                 descripcion="Agrega los días faltantes con la mediana de los k días anteriores del mismo mes y la columna imputado",
                 parametros={"k": k}, detalle="tratamientos_EDA.json > dias_faltantes")
    df = aplicar(log, df, agregar_calendario, etapa="Calendario",
                 descripcion="Día de la semana, mes, estación y fin de semana del día que se predice (D+1), con su codificación seno/coseno")
    if con_target:
        df = aplicar(log, df, agregar_target, etapa="Target",
                     descripcion="target = Frio (Kw) del día siguiente; se saca el último día (no tiene target)")
    return df


def construir_dataset(rutas, registro_eda, con_target=True, log=None):
    """Excel(s) crudo(s) -> una fila por día. Punto de entrada para train y predict."""
    return armar_dataset(cargar_excels(rutas), registro_eda, con_target, log)


def guardar_dataset(df, ruta):
    """Guarda el dataset diario como CSV (DIA en formato aaaa-mm-dd, sin índice),
    con el mismo formato que datos_base.csv del EDA. Devuelve la ruta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, date_format="%Y-%m-%d")
    return ruta


def tratar_outliers(datos, registro, log=None):
    """Las dos pasadas de z-score de la notebook, con los números de train del registro."""
    df = datos
    for pasada in ("pasada1", "pasada2"):
        df = aplicar(log, df, corregir_outliers, registro, pasada, etapa="Outliers",
                     descripcion="Marca |x - media| / desvío > umbral y lleva solo esas celdas al p2 / p98 de train",
                     parametros={"umbral": registro["outliers"]["umbral"],
                                 "columnas_analizadas": len(registro["outliers"][pasada]["media"])},
                     detalle=f"tratamientos_PREPROCESSING.json > outliers > {pasada}")
    return df


def crear_features(datos, registro, temp, log=None):
    """Feature engineering de la notebook, en el orden de la clave "features" del registro.
    Una feature se saltea si falta su columna de origen (el Excel de test puede traer menos columnas);
    si era una de las seleccionadas, `seleccionar` avisa cuál falta. Al final borra las metas."""
    f = registro["features"]
    etapa, fuente = "Feature engineering", "tratamientos_PREPROCESSING.json > features"
    df = aplicar(log, datos, agregar_temperatura, temp, etapa=etapa,
                 descripcion="Temperatura media, máxima y mínima del día (Open-Meteo, Tecate) y su variación",
                 parametros={"csv": f["temperatura"]["csv"]}, detalle=f"{fuente} > temperatura")

    for nombre, (columna, ventana) in f["medias_moviles"].items():
        if columna in df.columns:
            df = aplicar(log, df, media_movil, columna, nombre, ventana, etapa=etapa,
                         descripcion=f"{nombre}: media móvil de {ventana} días (incluye D)",
                         parametros={"columna": columna, "ventana": ventana}, detalle=f"{fuente} > medias_moviles")
    for nombre, columna in f["variaciones"].items():
        if columna in df.columns:
            df = aplicar(log, df, variacion, columna, nombre, etapa=etapa,
                         descripcion=f"{nombre}: diferencia contra el día anterior",
                         parametros={"columna": columna}, detalle=f"{fuente} > variaciones")
    for nombre, (columna, k) in f["rezagos"].items():
        if columna in df.columns:
            df = aplicar(log, df, rezago, columna, nombre, k, etapa=etapa,
                         descripcion=f"{nombre}: valor de {k} días antes",
                         parametros={"columna": columna, "k": k}, detalle=f"{fuente} > rezagos")

    ratios = {nombre: par for nombre, par in f["ratios"].items() if set(par) <= set(df.columns)}
    df = aplicar(log, df, agregar_combinaciones, ratios=ratios, etapa=etapa,
                 descripcion="Ratios producción / frío del mismo día (0 si el frío es 0)",
                 parametros={"ratios": len(ratios)}, detalle=f"{fuente} > ratios")
    df = aplicar(log, df, agregar_eficiencias, f["eficiencias"], registro["medias_train"]["eficiencias"], etapa=etapa,
                 descripcion="efic_<X> = KPI / meta en los días con meta; el resto, la media de train",
                 parametros={"pares": len(f["eficiencias"])},
                 detalle=f"{fuente} > eficiencias y tratamientos_PREPROCESSING.json > medias_train")

    return aplicar(log, df, borrar_columnas, registro, "metas_tras_eficiencias", etapa=etapa,
                   descripcion="Borra las metas (ya están dentro de las efic_*)",
                   parametros={"columnas_registradas": len(registro["metas_tras_eficiencias"])},
                   detalle="tratamientos_PREPROCESSING.json > metas_tras_eficiencias")


def preprocesar(diario, registro, temp, log=None):
    """Dataset diario (salida de construir_dataset) -> DIA + features seleccionadas (+ target).
    Aplica lo que decidió la notebook leyendo el registro: no ajusta nada con los datos que recibe."""
    df = tratar_outliers(diario, registro, log)
    df = crear_features(df, registro, temp, log)
    return aplicar(log, df, seleccionar, registro, etapa="Selección",
                   descripcion="Deja DIA, las features elegidas por LASSO y Random Forest (unión) y el target",
                   parametros={"features": len(registro["seleccion"]["features"])},
                   detalle="tratamientos_PREPROCESSING.json > seleccion")


def armar_ejecucion(dfs):
    """Encabezado de data_lineage.json: script, fecha y archivos de entrada."""
    return {
        "script": "src/preprocessing_pipeline.py",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "estado": "en curso",
        "entradas": {f"{nombre}.xlsx": {"filas": len(df), "columnas": df.shape[1]} for nombre, df in dfs.items()},
        "hojas_excluidas": sorted(HOJAS_EXCLUIDAS),
    }


def describir_salida(final, conjuntos):
    """Dataset final y shapes de X_train, X_val, X_test."""
    return {
        "archivo": "data/processed/dataset_final.csv", "filas": len(final),
        "columnas": final.shape[1], "escalado": False,
        "conjuntos": {nombre: {"X_shape": [len(df), df.shape[1] - 2],   # sin DIA ni target
                               "desde": str(df["DIA"].min().date()),
                               "hasta": str(df["DIA"].max().date())}
                      for nombre, df in conjuntos.items()},
    }


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent.parent
    proc = raiz / "data" / "processed"
    rutas = list((raiz / "data" / "raw").glob("Totalizadores Planta *.xlsx"))

    # Decisiones del EDA (falla si no existe o está vacío)
    registro_eda = leer_tratamiento(proc / "tratamientos_EDA.json")

    dfs = cargar_excels(rutas)
    for nombre, df in dfs.items():
        print(f"{nombre}: {df.shape[0]} filas x {df.shape[1]} cols")

    # data_lineage.json se rehace en cada corrida. Si algo falla, se guarda igual hasta el paso que falló
    pasos = []
    lineage = {
        "descripcion": "Transformaciones que aplica src/preprocessing_pipeline.py, en orden. "
                       "Cada paso indica en 'detalle' dónde están las columnas y parámetros completos.",
        "ejecucion": armar_ejecucion(dfs),
        "fuentes": {"eda": "data/processed/tratamientos_EDA.json",
                    "preprocesamiento": "data/processed/tratamientos_PREPROCESSING.json"},
        "pasos": pasos,
    }
    ruta_lineage = proc / "data_lineage.json"
    try:
        dataset = armar_dataset(dfs, registro_eda, log=pasos)
        print(f"Dataset diario: {dataset.shape} | {dataset['DIA'].min().date()} -> {dataset['DIA'].max().date()}")

        # Entrada de la notebook de preprocesamiento: limpieza del EDA aplicada, antes de outliers y features
        print("Guardado en", guardar_dataset(dataset, proc / "datos_preprocess.csv"))

        # Decisiones de la notebook de preprocesamiento: recién hacen falta acá
        registro = leer_tratamiento(proc / "tratamientos_PREPROCESSING.json")

        temp = leer_temperatura(raiz / registro["features"]["temperatura"]["csv"])
        final = preprocesar(dataset, registro, temp, log=pasos)

        conjuntos = separar_conjuntos(final, registro)
        chequear_conjuntos(conjuntos)
        for nombre, df in conjuntos.items():
            X = df.drop(columns=["DIA", "target"])
            print(f"X_{nombre}: {X.shape} | {df['DIA'].min().date()} -> {df['DIA'].max().date()}")

        print("Guardado en", guardar_dataset(final, proc / "dataset_final.csv"))
        lineage["salida"] = describir_salida(final, conjuntos)
        lineage["ejecucion"]["estado"] = "completo"
    except Exception as e:
        lineage["ejecucion"]["estado"] = f"error en el paso {len(pasos) + 1}: {type(e).__name__}: {e}"
        raise
    finally:
        ruta_lineage.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Registro de la corrida en {ruta_lineage} ({len(pasos)} pasos)")
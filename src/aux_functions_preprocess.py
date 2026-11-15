from datetime import datetime, time
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

# Lo mínimo que tiene que traer cualquier Excel: hoja -> columnas obligatorias.
# Cuando elijan las features del modelo, se agregan acá.
HOJAS_REQUERIDAS = {"Consolidado EE": ["Frio (Kw)"]}

# Hojas que el EDA descarta. Se excluyen por nombre (en el camino de predicción).
HOJAS_EXCLUIDAS = {"Auxiliar", "Seguimiento Dia"}

class ErrorEstructura(ValueError):
    """El Excel no tiene las hojas o columnas que el pipeline necesita."""


def normalizar_nombre(nombre):
    """' Día ' -> 'DIA'. Solo para comparar nombres; no renombra nada."""
    sin_tildes = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.split()).upper()


def validar_estructura(hojas, requeridas=None):
    """Revisa que estén las hojas y columnas requeridas ({hoja: [columnas]}).
    Sin `requeridas` usa HOJAS_REQUERIDAS (el mínimo, para entrenamiento).
    Junta todo lo que falta y falla una sola vez con la lista completa."""
    requeridas = requeridas or HOJAS_REQUERIDAS
    faltantes = []
    for hoja, columnas in requeridas.items():
        if hoja not in hojas:
            faltantes.append(f"  - falta la hoja '{hoja}'")
            continue
        presentes = {str(c).strip() for c in hojas[hoja].columns}
        faltantes += [f"  - falta la columna '{c}' en la hoja '{hoja}'"
                      for c in columnas if c not in presentes]
    if faltantes:
        raise ErrorEstructura(
            "El Excel no tiene la estructura esperada:\n" + "\n".join(faltantes)
            + f"\nHojas encontradas: {sorted(hojas)}"
        )


def buscar_columnas_clave(df, hoja):
    """Busca DIA y HORA por nombre (sin importar tildes, espacios ni mayúsculas).
    Devuelve {"DIA": nombre_real, "HORA": nombre_real}.
    Falla si falta alguna o si hay más de una candidata."""
    encontradas, faltan = {}, []
    for clave in ("DIA", "HORA"):
        candidatas = [c for c in df.columns if normalizar_nombre(c) == clave]
        if not candidatas:
            faltan.append(clave)
        elif len(candidatas) > 1:
            raise ErrorEstructura(
                f"Hoja '{hoja}': hay {len(candidatas)} columnas que pueden ser {clave}: {candidatas}"
            )
        else:
            encontradas[clave] = candidatas[0]
    if faltan:
        raise ErrorEstructura(
            f"Hoja '{hoja}': no se encontró la columna {' ni '.join(faltan)}. "
            f"Primeras columnas de la hoja: {list(df.columns[:5])}"
        )
    return encontradas


def limpiar_dias(serie):
    """'1/7/2020  00:00:00' -> 1/7/2020, vectorizado para toda la columna."""
    texto = (
        serie.astype("string")
        .str.strip()
        .str.replace(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", regex=True)  # el "  00:00:00" del final
    )
    # Caso principal: texto día/mes/año
    fechas = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    # Por si alguna celda Excel ya la guardaba como fecha: al pasarla a texto
    # queda "2020-07-01", que no matchea el formato de arriba
    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    fechas = fechas.fillna(iso)
    # Los "0" y basura quedan NaT o caen en 1899/1900 -> se descartan
    return fechas.where(fechas.dt.year >= 2000)


def normalizar_hora(serie):
    """Deja todas las horas como texto 'HH:MM:SS'.
    La mayoría ya viene así, pero algunas hojas guardan la hora como fecha+hora
    (1900-01-01 01:00:00) y no coincidirían con las demás al unir por DIA + HORA."""
    def a_texto(h):
        if isinstance(h, (datetime, time)):
            return h.strftime("%H:%M:%S")
        return str(h).strip()
    return serie.map(a_texto)


def procesar_hoja(df, hoja):
    """Busca DIA y HORA por nombre y recorta las filas sin clave válida.
    Devuelve (df_limpio, filas_eliminadas)."""
    df = df.set_axis([str(c).strip() for c in df.columns], axis=1)
    claves = buscar_columnas_clave(df, hoja)
    df = df.rename(columns={claves["DIA"]: "DIA", claves["HORA"]: "HORA"})

    dias = limpiar_dias(df["DIA"])
    mask = dias.notna() & df["HORA"].notna()
    out = df.loc[mask].copy()
    out["DIA"] = dias[mask]
    out["HORA"] = normalizar_hora(out["HORA"])
    return out.reset_index(drop=True), int((~mask).sum())


def unir_hojas(hojas_limpias):
    """Une las hojas por DIA + HORA sin descartar filas repetidas.
    Deja las filas de cada día en el orden original del Excel."""
    tablas = []
    for nombre, df in hojas_limpias.items():
        df = df.copy()
        # 0 = primera aparición de ese DIA+HORA, 1 = segunda, ...
        # Se usa solo para unir: sin esto el índice tiene claves repetidas y el concat falla
        df["N_REPETICION"] = df.groupby(["DIA", "HORA"], dropna=False).cumcount()
        claves = ["DIA", "HORA", "N_REPETICION"]
        tablas.append(df.set_index(claves).add_prefix(f"{nombre} | "))
    unido = pd.concat(tablas, axis=1, join="outer").reset_index()
    # Ordena solo por día; "stable" no cambia el orden de las filas dentro de cada día
    unido = unido.sort_values("DIA", kind="stable").reset_index(drop=True)
    return unido.drop(columns="N_REPETICION")


def nombre_salida(archivo):
    m = re.search(r"(\d{4})_(\d{4})", archivo.stem)  # el "2021_2022" del nombre
    sufijo = f"{m.group(1)}_{m.group(2)}" if m else archivo.stem
    return f"Totalizador_Formateado {sufijo}"

def a_segundos(horas):
    """Serie de horas en texto ('23:59:00') -> segundos desde las 00:00. NaN si no se entiende."""
    return pd.to_timedelta(horas, errors="coerce").dt.total_seconds()


def a_float(serie):
    """Texto con coma decimal / punto de miles -> float. Lo que no se entiende queda NaN."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype("float64")
    txt = serie.astype("string").str.strip()
    # punto de miles: hay 2 o más puntos, o hay punto y coma a la vez
    miles = (txt.str.count(r"\.").ge(2) | (txt.str.contains(".", regex=False) & txt.str.contains(",", regex=False))).fillna(False)
    txt = txt.mask(miles, txt.str.replace(".", "", regex=False))
    return pd.to_numeric(txt.str.replace(",", ".", regex=False), errors="coerce").astype("float64")


def corregir_dias_pegados(df, frio="Consolidado EE | Frio (Kw)"):
    """Días con más de 26 lecturas (varios días pegados bajo la misma fecha), EDA §7.3.
    1. Corta en bloques cuando la hora vuelve para atrás y arranca en 00:xx.
    2. Si el Frio de las 23:59 del bloque k es igual al del día DIA + k, el bloque es una copia y se saca.
    3. Recoloca fechas: cada (DIA, bloque) va a max(DIA, anterior + 1)."""
    max_filas = 26
    inicio_2359 = 23 * 3600 + 59 * 60

    df = df.reset_index(drop=True)
    conteo = df.groupby("DIA").size()
    es_pegado = df["DIA"].isin(conteo[conteo > max_filas].index)

    # 1. Bloques. Las filas sueltas del final del día (partidas) no abren un bloque.
    seg = a_segundos(df["HORA"])
    corte = (seg.groupby(df["DIA"]).diff() < 0) & (seg < 3600) & es_pegado
    bloque = corte.groupby(df["DIA"]).cumsum()

    # 2. Desempate: Frio de las 23:59 de cada bloque contra el del día DIA + bloque
    tmp = pd.DataFrame({"DIA": df["DIA"], "bloque": bloque, "frio": a_float(df[frio])})
    cierre = (tmp[seg.between(inicio_2359, inicio_2359 + 59)]
              .groupby(["DIA", "bloque"])["frio"].first().reset_index())
    base = cierre[cierre["bloque"] == 0].set_index("DIA")["frio"]
    extra = cierre[cierre["bloque"] > 0].reset_index(drop=True)
    frio_siguiente = base.reindex(extra["DIA"] + pd.to_timedelta(extra["bloque"], unit="D")).reset_index(drop=True)
    copias = extra[(extra["frio"] - frio_siguiente).abs() < 1e-6]
    copias = set(zip(copias["DIA"], copias["bloque"]))

    claves = list(zip(df["DIA"], bloque))
    queda = [k not in copias for k in claves]
    df = df[queda].copy()
    claves = [k for k, q in zip(claves, queda) if q]

    # 3. Fechas: los días faltantes (31, 29-31/08) absorben el corrimiento
    nueva_fecha, ultima = {}, None
    for dia, b in sorted(set(claves)):
        ultima = dia if ultima is None else max(dia, ultima + pd.Timedelta(days=1))
        nueva_fecha[(dia, b)] = ultima
    df["DIA"] = [nueva_fecha[k] for k in claves]
    return df.reset_index(drop=True)


def recortar_julio_agosto(df):
    """Archivos que arrancan el 01/07 y no son el primero: no conocen el corrimiento
    del archivo anterior, así que jul-ago de su primer año se toman del anterior."""
    inicio = pd.Timestamp(year=df["DIA"].min().year, month=9, day=1)
    return df[df["DIA"] >= inicio].copy()


def sacar_dias_cortos(df):
    """Saca los días con menos de 20 lecturas (EDA §7.4)."""
    min_lecturas = 20
    lecturas = df.groupby("DIA")["DIA"].transform("size")
    return df[lecturas >= min_lecturas].copy()


def filtrar_2359(df):
    """Deja solo las filas con hora 23:59:xx (23:59:00, 23:59:01, 23:59:59)."""
    seg = a_segundos(df["HORA"].astype("string"))
    inicio = 23 * 3600 + 59 * 60          # 23:59:00
    fin = inicio + 59                     # 23:59:59
    return df[(seg >= inicio) & (seg <= fin)].copy()


def convertir_a_float(df):
    """Pasa a float todas las columnas menos DIA y HORA (EDA §7.7)."""
    df = df.copy()
    cols = df.columns.drop(["DIA", "HORA"], errors="ignore")
    df[cols] = df[cols].apply(a_float)
    # La asignación deja un bloque por columna (PerformanceWarning en los pasos
    # siguientes); copy() junta todo de nuevo. No cambia los datos.
    df[cols] = df[cols].apply(a_float)
    return df


def unir_archivos(dfs):
    """Une los archivos (ya filtrados a 23:59), saca HORA, ARCHIVO y las Unnamed,
    y deja una fila por día (EDA §7.8, §7.9, §7.10):
    dentro de cada día gana la fila con menos nulos (si empatan, la que venía primero)
    y sus huecos se rellenan con las otras filas del mismo día."""
    partes = []
    for nombre, df in dfs.items():
        f = df.copy()
        f.insert(0, "ARCHIVO", nombre)
        partes.append(f)
    total = (pd.concat(partes, ignore_index=True)
             .sort_values(["DIA", "HORA"], kind="stable").reset_index(drop=True))

    total = total.drop(columns=["HORA", "ARCHIVO"])
    total = total.drop(columns=[c for c in total.columns if "Unnamed" in c])
    # El concat de archivos con columnas distintas deja el df en cientos de bloques
    # (PerformanceWarning en todos los pasos siguientes); copy() lo junta. No cambia los datos.
    total = total.copy()

    cols_datos = total.columns.drop("DIA")
    total["_nulos"] = total[cols_datos].isna().sum(axis=1)
    total["_orden"] = range(len(total))
    ordenado = total.sort_values(["DIA", "_nulos", "_orden"])
    # first() toma, por columna, el primer valor no nulo del grupo
    unido = ordenado.groupby("DIA", sort=True)[list(cols_datos)].first().reset_index()
    assert unido["DIA"].is_unique
    return unido


def rellenar_dias_faltantes(df, k=10):
    """Días faltantes entre el mínimo y el máximo: mediana de los k días anteriores
    del mismo mes (regla del EDA). Agrega `imputado` (True en los días rellenados)."""
    observados = df.set_index("DIA").sort_index()
    rango = pd.date_range(observados.index.min(), observados.index.max(), freq="D")
    faltantes = rango.difference(observados.index)
    base = observados.reindex(rango).rename_axis("DIA")
    for f in faltantes:
        previos = observados[(observados.index < f) & (observados.index.month == f.month)].tail(k)
        base.loc[f] = previos.median()
    # Asignar filas enteras con .loc deja el df en cientos de bloques
    # (PerformanceWarning acá y en agregar_target); copy() lo junta. No cambia los datos.
    base = base.copy()
    base["imputado"] = base.index.isin(faltantes)
    return base.reset_index()


def agregar_calendario(diario, col_fecha="DIA"):
    """Agrega variables de calendario del día que se predice (D+1).

    DIA_SEMANA: 0 = lunes ... 6 = domingo
    MES:        1 = enero ... 12 = diciembre
    ESTACION:   0 = invierno (dic-feb), 1 = primavera (mar-may),
                2 = verano (jun-ago), 3 = otoño (sep-nov)  [hemisferio norte]
    DIA_SEMANA y MES llevan codificación cíclica (_sin/_cos). La estación va en
    one-hot con 4 columnas fijas (EST_invierno ... EST_otono), aunque el lote
    traiga una sola estación.
    """
    df = diario.copy()
    fecha_objetivo = df[col_fecha] + pd.Timedelta(days=1)

    df["DIA_SEMANA"] = fecha_objetivo.dt.dayofweek
    df["MES"] = fecha_objetivo.dt.month
    df["ESTACION"] = (df["MES"] % 12) // 3
    df["FIN_SEMANA"] = (df["DIA_SEMANA"] >= 5).astype(int)

    ciclos = [("DS", df["DIA_SEMANA"], 7), ("MES", df["MES"] - 1, 12)]
    for prefijo, valor, periodo in ciclos:
        df[f"{prefijo}_sin"] = np.sin(2 * np.pi * valor / periodo)
        df[f"{prefijo}_cos"] = np.cos(2 * np.pi * valor / periodo)

    for i, nombre in enumerate(["invierno", "primavera", "verano", "otono"]):
        df[f"EST_{nombre}"] = (df["ESTACION"] == i).astype(int)

    for i, nombre in enumerate(["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]):
        df[f"DS_{nombre}"] = (df["DIA_SEMANA"] == i).astype(int)

    return df
    

def agregar_target(df, frio="Consolidado EE | Frio (Kw)"):
    """target = Frio (Kw) del día siguiente. Saca las filas sin target (EDA §7.13)."""
    df = df.sort_values("DIA").reset_index(drop=True)
    df["target"] = df[frio].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def leer_tratamiento(ruta):
    """Lee un JSON de tratamientos (tratamientos_EDA.json o tratamientos_PREPROCESSING.json).
    Falla si no existe o está vacío: se generan corriendo la notebook correspondiente."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}. Se genera corriendo su notebook (Run All).")
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    if not registro:
        raise ValueError(f"{ruta} está vacío. Correr su notebook completa (Run All) antes del pipeline.")
    return registro


def celdas_modificadas(antes, despues, col_fecha="DIA"):
    """Cantidad de celdas que cambiaron de valor entre dos versiones del df,
    comparando por fecha y en las columnas numéricas que tienen las dos.
    None si no se puede comparar fila a fila (fechas repetidas, antes de quedar un día por fila)."""
    if col_fecha not in antes or col_fecha not in despues:
        return None
    if not (antes[col_fecha].is_unique and despues[col_fecha].is_unique):
        return None
    a, b = antes.set_index(col_fecha), despues.set_index(col_fecha)
    filas = a.index.intersection(b.index)
    cols = [c for c in a.columns.intersection(b.columns)
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    x = a.loc[filas, cols].to_numpy(dtype=float)
    y = b.loc[filas, cols].to_numpy(dtype=float)
    return int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())


def registrar_paso(log, antes, despues, etapa, descripcion, funcion=None, parametros=None, detalle=None):
    """Agrega a `log` (lista) una entrada legible de un paso: qué hizo, con qué parámetros,
    dónde está el detalle y cómo cambió el df. Con log=None no hace nada (predict.py)."""
    if log is None:
        return
    cols_antes, cols_despues = set(antes.columns), set(despues.columns)
    entrada = {
        "paso": len(log) + 1,
        "etapa": etapa,
        "funcion": funcion,
        "descripcion": descripcion,
        "parametros": parametros or {},
        "filas": [len(antes), len(despues)],
        "columnas": [antes.shape[1], despues.shape[1]],
        "columnas_agregadas": sorted(cols_despues - cols_antes),
        "columnas_borradas": len(cols_antes - cols_despues),
        "nulos": [int(antes.isna().sum().sum()), int(despues.isna().sum().sum())],
        "celdas_modificadas": celdas_modificadas(antes, despues),
    }
    if detalle:
        entrada["detalle"] = detalle
    log.append(entrada)


def aplicar(log, df, funcion, *args, etapa, descripcion, parametros=None, detalle=None, **kwargs):
    """Aplica funcion(df, *args, **kwargs), registra el paso en `log` y devuelve el resultado."""
    nuevo = funcion(df, *args, **kwargs)
    registrar_paso(log, df, nuevo, etapa, descripcion, funcion.__name__, parametros, detalle)
    return nuevo


def columnas_meta(datos):
    """Columnas de metas: las que empiezan con 'Meta' después del nombre de la hoja."""
    return [c for c in datos.columns if c.split(" | ", 1)[-1].strip().lower().startswith("meta")]


def columnas_con_tratamiento(datos, registro, tratamiento):
    """Columnas de `datos` que tienen ese tratamiento registrado en el JSON."""
    return [c for c, lista in registro.get("tratamientos", {}).items()
            if tratamiento in lista and c in datos.columns]


def borrar_columnas(datos, registro, paso):
    """Borra las columnas que la notebook registró en `paso`.
    errors="ignore": el Excel de test puede no traer alguna de esas columnas."""
    return datos.drop(columns=registro[paso], errors="ignore")


def marcar_sin_meta(datos):
    """Agrega no_hay_meta (1 si el día no tiene ninguna meta) y rellena las metas nulas con 1.
    El 1 es provisorio: las eficiencias se calculan solo en días con meta."""
    df = datos.copy()
    metas = columnas_meta(df)
    df["no_hay_meta"] = df[metas].isna().all(axis=1).astype(int)
    df[metas] = df[metas].fillna(1)
    return df


def imputar_promedio_anteriores(datos, columnas, n=2, col_fecha="DIA", invalidar_negativos=False):
    """Rellena cada nulo con el promedio de los `n` valores válidos anteriores.

    Solo mira hacia atrás: no usa información del futuro
    y se comporta igual en train que en predict.py.
    Si hay varios nulos seguidos, todos reciben el promedio de los n válidos previos al hueco.
    Si no hay ningún valor válido antes (inicio de la serie), usa el siguiente válido.
    Si `invalidar_negativos=True`, los negativos se tratan como nulos.
    """
    df = datos.sort_values(col_fecha).copy()
    if invalidar_negativos:
        df[columnas] = df[columnas].mask(df[columnas] < 0)
    for col in columnas:
        s = df[col]
        media_previa = s.dropna().rolling(n, min_periods=1).mean().reindex(s.index).ffill()
        df[col] = s.fillna(media_previa).fillna(s.bfill())
    return df


def linea_de(col):
    """Devuelve el área (Cocina, Línea 2-5, Filtro) según el nombre de la columna."""
    nombre = col.split(" | ", 1)[-1]
    patrones = [("Cocina", r"Cocina"), ("Filtro", r"Filtro")] + \
               [(f"Línea {n}", rf"L(?:inea)?\s*{n}\b") for n in [2, 3, 4, 5]]
    for area, patron in patrones:
        if re.search(patron, nombre, flags=re.IGNORECASE):
            return area
    return "Otra"


def marcar_nulos_por_area(datos, indicadores):
    """Agrega nulo_<área>: 1 si ese día faltaba algún dato del área (antes de imputar).
    `indicadores`: {nombre: [columnas]}, clave "indicadores_nulos" del JSON del EDA."""
    df = datos.copy()
    for nombre, columnas in indicadores.items():
        cols = [c for c in columnas if c in df.columns]
        df[nombre] = df[cols].isna().any(axis=1).astype(int)
    return df


def separar_conjuntos(datos, registro, col_fecha="DIA"):
    """Corta en train / val / test con las fechas que guardó la notebook (clave "split").
    No recalcula el corte: usa fin_train y fin_val del registro."""
    fin_train = pd.Timestamp(registro["split"]["fin_train"])
    fin_val = pd.Timestamp(registro["split"]["fin_val"])
    fechas = datos[col_fecha]
    partes = {"train": fechas <= fin_train,
              "val": (fechas > fin_train) & (fechas <= fin_val),
              "test": fechas > fin_val}
    return {nombre: datos[mascara].reset_index(drop=True) for nombre, mascara in partes.items()}


def corregir_outliers(datos, registro, pasada, frio="Consolidado EE | Frio (Kw)", col_fecha="DIA"):
    """Una pasada de outliers con los números de train guardados en el registro:
    marca |x - media| / desvío > umbral y lleva SOLO esas celdas a p_inf / p_sup.
    Si se corrige Frio y hay target, lo recalcula (Frio de D+1)."""
    parametros = registro["outliers"][pasada]
    cols = [c for c in parametros["media"] if c in datos.columns]
    media, desvio, p_inf, p_sup = (pd.Series(parametros[k])[cols] for k in ("media", "desvio", "p_inf", "p_sup"))

    x = datos[cols]
    mascara = ((x - media) / desvio).abs() > registro["outliers"]["umbral"]
    recortado = x.clip(lower=p_inf, upper=p_sup, axis=1)

    df = datos.copy()
    df[cols] = x.where(~mascara, recortado)

    if "target" in df.columns and frio in cols and mascara[frio].any():
        frio_nuevo = df.set_index(col_fecha)[frio]
        df["target"] = (df[col_fecha] + pd.Timedelta(days=1)).map(frio_nuevo).fillna(df["target"])
    return df


def leer_temperatura(ruta):
    """Lee el CSV de temperatura diaria de Tecate (bajado de Open-Meteo en la notebook)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de temperatura: {ruta}")
    return pd.read_csv(ruta, parse_dates=["DIA"])


def agregar_temperatura(datos, temp, col_fecha="DIA"):
    """Pega la temperatura del día D y su variación contra el día anterior (> 0: está subiendo).
    La variación se calcula sobre la tabla de temperatura (calendario continuo),
    así que funciona igual en cualquier conjunto aunque se procese por separado."""
    temp = temp.sort_values(col_fecha).copy()
    temp["TEMP_dif"] = temp["TEMP_media"].diff()
    return datos.merge(temp, on=col_fecha, how="left", validate="m:1")


def serie_con_previo(datos, columna, previo=None, col_fecha="DIA"):
    """Serie de `columna` indexada por fecha, con los días de `previo` adelante (si se pasa)."""
    partes = [datos[[col_fecha, columna]]] if previo is None else [previo[[col_fecha, columna]], datos[[col_fecha, columna]]]
    return pd.concat(partes).sort_values(col_fecha).set_index(col_fecha)[columna]


def media_movil(datos, columna, nombre, ventana, previo=None, col_fecha="DIA"):
    """Promedio de los últimos `ventana` días, incluyendo el día D (no usa datos futuros).
    `previo`: días anteriores a `datos`, para que los primeros días tengan historia."""
    resultado = serie_con_previo(datos, columna, previo, col_fecha).rolling(ventana, min_periods=1).mean()
    datos = datos.copy()
    datos[nombre] = datos[col_fecha].map(resultado)
    return datos


def variacion(datos, columna, nombre, previo=None, col_fecha="DIA"):
    """Diferencia contra el día anterior (> 0: está creciendo).
    Si no hay día anterior (inicio de la serie, sin `previo`), queda en 0."""
    resultado = serie_con_previo(datos, columna, previo, col_fecha).diff().fillna(0)
    datos = datos.copy()
    datos[nombre] = datos[col_fecha].map(resultado)
    return datos


def rezago(datos, columna, nombre, k, previo=None, col_fecha="DIA"):
    """Valor de `columna` k días antes de D. Con `previo`, los primeros días toman la historia.
    Si no hay k días anteriores (inicio de la serie), usa el primer valor disponible."""
    resultado = serie_con_previo(datos, columna, previo, col_fecha).shift(k).bfill()
    datos = datos.copy()
    datos[nombre] = datos[col_fecha].map(resultado)
    return datos


def media_mismo_dia(datos, columna, nombre, semanas, previo=None, col_fecha="DIA"):
    """Promedio de `columna` en los últimos `semanas` días con el mismo día de la semana que D+1
    (D-6, D-13, D-20, ...). Al inicio de la serie promedia los que haya; si no hay ninguno,
    usa el primer valor disponible (como rezago)."""
    serie = serie_con_previo(datos, columna, previo, col_fecha)
    mismos_dias = pd.concat([serie.shift(6 + 7 * s) for s in range(semanas)], axis=1)
    resultado = mismos_dias.mean(axis=1).bfill()
    datos = datos.copy()
    datos[nombre] = datos[col_fecha].map(resultado)
    return datos


def agregar_combinaciones(datos, ratios=None, interacciones=None):
    """Agrega columnas calculadas con valores del mismo día (no ajusta nada con train, no necesita `previo`).
    ratios: {nombre: (numerador, denominador)} -> numerador / denominador (0 si el denominador es 0).
    interacciones: {nombre: (col_a, col_b)} -> col_a * col_b."""
    ratios, interacciones = ratios or {}, interacciones or {}
    nuevas = {}
    for nombre, (num, den) in ratios.items():
        nuevas[nombre] = (datos[num] / datos[den]).where(datos[den] != 0, 0)
    for nombre, (a, b) in interacciones.items():
        nuevas[nombre] = datos[a] * datos[b]
    datos = datos.drop(columns=list(nuevas), errors="ignore")   # si se vuelve a correr, no duplica
    return pd.concat([datos, pd.DataFrame(nuevas, index=datos.index)], axis=1)


def agregar_eficiencias(datos, pares, medias):
    """Crea efic_<X> = KPI / meta para cada par {nombre: [meta, kpi]} del registro.
    Solo en días con meta (no_hay_meta == 0) y meta distinta de 0; el resto lleva la media de train."""
    nuevas = {}
    for nombre, (meta, kpi) in pares.items():
        if meta not in datos.columns or kpi not in datos.columns:
            continue
        efic = (datos[kpi] / datos[meta]).where(datos["no_hay_meta"].eq(0) & datos[meta].ne(0))
        nuevas[nombre] = efic.fillna(medias[nombre])
    datos = datos.drop(columns=list(nuevas), errors="ignore")
    return pd.concat([datos, pd.DataFrame(nuevas, index=datos.index)], axis=1)


def columnas_origen(feature, registro):
    """Columnas del Excel de las que sale una feature, según la clave "features" del registro.
    Una columna cruda (o de calendario) se devuelve a sí misma."""
    f = registro["features"]
    if feature in f["medias_moviles"]:
        origen = [f["medias_moviles"][feature][0]]
    elif feature in f["rezagos"]:
        origen = [f["rezagos"][feature][0]]
    elif feature in f["medias_mismo_dia"]:
        origen = [f["medias_mismo_dia"][feature][0]]
    elif feature in f["variaciones"]:
        origen = [f["variaciones"][feature]]
    elif feature in f["ratios"]:
        origen = list(f["ratios"][feature])
    elif feature in f["eficiencias"]:
        origen = list(f["eficiencias"][feature])
    elif feature in f["temperatura"]["columnas"]:
        return [f["temperatura"]["csv"]]
    else:
        return [feature]
    # FRIO_media5_dif sale de FRIO_media5, que sale de Frio: se sigue hasta la columna cruda
    return [c for o in origen for c in columnas_origen(o, registro)]


def columnas_requeridas(features, registro, frio="Consolidado EE | Frio (Kw)"):
    """Hojas y columnas del Excel que hacen falta para calcular `features`: {hoja: [columnas]}.
    Sigue cada feature hasta sus columnas crudas con columnas_origen. Las que no tienen
    "Hoja | " (calendario, temperatura, indicadores) las crea el pipeline y no se piden.
    Frio se pide siempre (días pegados y features de frío)."""
    origenes = {frio} | {c for f in features for c in columnas_origen(f, registro)}
    requeridas = {}
    for col in sorted(origenes):
        if " | " in col:
            hoja, columna = col.split(" | ", 1)
            requeridas.setdefault(hoja, []).append(columna)
    return requeridas


def seleccionar(datos, registro, col_fecha="DIA"):
    """Deja DIA + las features elegidas en la notebook (+ target si está).
    Si falta alguna, falla diciendo cuáles y de qué columnas del Excel salen."""
    features = registro["seleccion"]["features"]
    faltan = [c for c in features if c not in datos.columns]
    if faltan:
        lineas = []
        for c in faltan:
            origen = columnas_origen(c, registro)
            lineas.append(f"  - {c}" if origen == [c] else f"  - {c} (se calcula de: {', '.join(origen)})")
        raise ErrorEstructura("Faltan features que el modelo necesita:\n" + "\n".join(lineas))
    final = [col_fecha, *features] + (["target"] if "target" in datos.columns else [])
    return datos[final].copy()


def chequear_conjuntos(conjuntos, col_fecha="DIA"):
    """Chequeo final: mismas columnas en los tres conjuntos, sin nulos ni infinitos
    y solo la fecha como columna no numérica. Falla con la lista de problemas."""
    problemas = []
    columnas = list(conjuntos["train"].columns)
    for nombre, df in conjuntos.items():
        if list(df.columns) != columnas:
            problemas.append(f"{nombre}: no tiene las mismas columnas que train")
        if len(df) == 0:
            problemas.append(f"{nombre}: no tiene filas")
        no_numericas = [c for c in df.columns if c != col_fecha and not pd.api.types.is_numeric_dtype(df[c])]
        if no_numericas:
            problemas.append(f"{nombre}: columnas no numéricas: {no_numericas}")
        numericas = df.select_dtypes("number")
        con_nulos = list(numericas.columns[numericas.isna().any()])
        if con_nulos:
            problemas.append(f"{nombre}: columnas con nulos: {con_nulos}")
        con_inf = list(numericas.columns[np.isinf(numericas).any()])
        if con_inf:
            problemas.append(f"{nombre}: columnas con infinitos: {con_inf}")
    if problemas:
        raise ValueError("El dataset final no pasa el chequeo:\n" + "\n".join(f"  - {p}" for p in problemas))
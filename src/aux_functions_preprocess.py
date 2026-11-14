from datetime import datetime, time
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


def validar_estructura(hojas):
    """Revisa que estén las hojas y columnas de HOJAS_REQUERIDAS.
    Junta todo lo que falta y falla una sola vez con la lista completa."""
    faltantes = []
    for hoja, columnas in HOJAS_REQUERIDAS.items():
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


def rellenar_dias_faltantes(df):
    """Días faltantes entre el mínimo y el máximo: promedio de los 2 días anteriores
    y los 2 posteriores (EDA §7.12)."""
    rango = pd.date_range(start=df["DIA"].min(), end=df["DIA"].max(), freq="D")
    faltantes = rango.difference(df["DIA"].dt.normalize())
    df = df.set_index("DIA").reindex(rango).rename_axis("DIA")
    vecinos = pd.concat([df.shift(k) for k in (2, 1, -1, -2)])
    promedio = vecinos.groupby(level=0).mean(numeric_only=True)
    df.loc[faltantes] = promedio.loc[faltantes]
    return df.reset_index()


def agregar_calendario(diario, col_fecha="DIA"):
    """Agrega variables de calendario del día que se predice (D+1).

    DIA_SEMANA: 0 = lunes ... 6 = domingo
    MES:        1 = enero ... 12 = diciembre
    ESTACION:   0 = invierno (dic-feb), 1 = primavera (mar-may),
                2 = verano (jun-ago), 3 = otoño (sep-nov)  [hemisferio norte]
    Las versiones _sin/_cos son la codificación cíclica de cada una.
    """
    df = diario.copy()
    fecha_objetivo = df[col_fecha] + pd.Timedelta(days=1)

    df["DIA_SEMANA"] = fecha_objetivo.dt.dayofweek
    df["MES"] = fecha_objetivo.dt.month
    df["ESTACION"] = (df["MES"] % 12) // 3
    df["FIN_SEMANA"] = (df["DIA_SEMANA"] >= 5).astype(int)

    ciclos = [("DS", df["DIA_SEMANA"], 7), ("MES", df["MES"] - 1, 12), ("EST", df["ESTACION"], 4)]
    for prefijo, valor, periodo in ciclos:
        df[f"{prefijo}_sin"] = np.sin(2 * np.pi * valor / periodo)
        df[f"{prefijo}_cos"] = np.cos(2 * np.pi * valor / periodo)

    return df
    

def agregar_target(df, frio="Consolidado EE | Frio (Kw)"):
    """target = Frio (Kw) del día siguiente. Saca las filas sin target (EDA §7.13)."""
    df = df.sort_values("DIA").reset_index(drop=True)
    df["target"] = df[frio].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)
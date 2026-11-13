from datetime import datetime, time
import re
import unicodedata

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
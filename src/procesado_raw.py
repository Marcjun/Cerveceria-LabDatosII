from datetime import datetime, time
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR  

# Solo agarra los originales; los "Totalizador_Formateado" no coinciden,
# así que correrlo dos veces no reprocesa sus propias salidas.
archivos = sorted(RAW_DIR.glob("Totalizadores Planta *.xlsx"))

AÑO_MIN_VALIDO = 2000
COLS_CLAVE = ["DIA", "HORA"]  
RE_HORA = r"\s+\d{1,2}:\d{2}(:\d{2})?$"   # el "  00:00:00" del final
RE_AÑOS = re.compile(r"(\d{4})_(\d{4})")  # el "2021_2022" del nombre


def limpiar_dias(serie):
    # Optimización de claude: vectorizó todo para que dure menos el borrado
    # de las columnas
    """'1/7/2020  00:00:00' -> 1/7/2020, vectorizado para toda la columna."""
    texto = (
        serie.astype("string")
        .str.strip()
        .str.replace(RE_HORA, "", regex=True)
    )
    # Caso principal: texto día/mes/año
    fechas = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    # Por si alguna celda Excel ya la guardaba como fecha: al pasarla a texto
    # queda "2020-07-01", que no matchea el formato de arriba
    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    fechas = fechas.fillna(iso)
    # Los "0" y basura quedan NaT o caen en 1899/1900 -> se descartan
    return fechas.where(fechas.dt.year >= AÑO_MIN_VALIDO)


def normalizar_hora(serie):
    """Deja todas las horas como texto 'HH:MM:SS'.
    La mayoría ya viene así, pero algunas hojas guardan la hora como fecha+hora
    (1900-01-01 01:00:00) y no coincidirían con las demás al unir por DIA + HORA."""
    def a_texto(h):
        if isinstance(h, (datetime, time)):
            return h.strftime("%H:%M:%S")
        return str(h).strip()
    return serie.map(a_texto)


def procesar_hoja(df):
    """Col 0 = DIA, col 1 = HORA. Recorta las filas sin clave válida.
    Devuelve (df_limpio, filas_eliminadas)."""
    cols = [str(c).strip() for c in df.columns]
    # Si otra columna ya se llama DIA u HORA, la renombro para que no choque
    cols = [f"{c}_orig" if i > 1 and c in COLS_CLAVE else c for i, c in enumerate(cols)]
    cols[0], cols[1] = "DIA", "HORA"
    df = df.set_axis(cols, axis=1)

    dias = limpiar_dias(df.iloc[:, 0])
    mask = dias.notna() & df.iloc[:, 1].notna()
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
        df["N_REPETICION"] = df.groupby(COLS_CLAVE, dropna=False).cumcount()
        claves = COLS_CLAVE + ["N_REPETICION"]
        tablas.append(df.set_index(claves).add_prefix(f"{nombre} | "))
    unido = pd.concat(tablas, axis=1, join="outer").reset_index()
    # Ordena solo por día; "stable" no cambia el orden de las filas dentro de cada día
    unido = unido.sort_values("DIA", kind="stable").reset_index(drop=True)
    return unido.drop(columns="N_REPETICION")


def nombre_salida(archivo):
    m = RE_AÑOS.search(archivo.stem)
    sufijo = f"{m.group(1)}_{m.group(2)}" if m else archivo.stem
    return f"Totalizador_Formateado {sufijo}"


for archivo in archivos:
    print(f"\n=== {archivo.name} ===")
    hojas = pd.read_excel(archivo, sheet_name=None, engine="calamine")

    limpias = {}
    for nombre, df in hojas.items():
        df_limpio, eliminadas = procesar_hoja(df)
        limpias[nombre] = df_limpio

    base = nombre_salida(archivo)

    # CSV con todas las hojas unidas
    unido = unir_hojas(limpias)
    unido.to_csv(OUT_DIR / f"{base}.csv", index=False, date_format="%d/%m/%Y", lineterminator="\n")
    print(f"  -> {base}.csv ({unido.shape[0]} filas x {unido.shape[1]} cols)")
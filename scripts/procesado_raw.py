from datetime import date, datetime
from pathlib import Path
import re

import pandas as pd

# Puedo ver que en los 3 excels hay varios datos que se van de la clave
# del día. Voy a recortarlos.

# Busco los paths de cada excel
# __file__ es donde está el script. 
# .resolve() hace que la ubicación del script sea el path completo
# .parent.parent sube dos niveles, o sea va del script a scripts, y 
# despues al directorio del proyecto. 
ROOT = Path(__file__).resolve().parent.parent
# Concatenas al directorio del proyecto el path de los excels del principio
RAW_DIR = ROOT / "data" / "raw"
# A donde termina mandando las cosas
OUT_DIR = ROOT / "data" / "excels"

# glob(coso) busca el coso dentro de la carpeta dada. Acá le doy 
# el totalizador planta entonces me va a buscar los 3 excels y los 
# ordena alfabeticamente (me di cuenta que es al pedo despues)
archivos = sorted(RAW_DIR.glob("Totalizadores Planta *.xlsx"))

COL_DIA = "DIA"
# un "0" de Excel como fecha cae en 1899/1900
AÑO_MIN_VALIDO = 2000  
# Voy a aprovechar limpiar ya la columna día de esas horas que tiene al lado, que no sirven
# de nada.
RE_HORA = re.compile(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*$")  
# PAra tomar el " 00:00:00" al final


# Esto limpia una única fila; 
def limpiar_valor_dia(v):
    """Devuelve la fecha sin hora, o None si la fila no tiene día asignado."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None

    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date() if v.year >= AÑO_MIN_VALIDO else None

    if isinstance(v, date):
        return v if v.year >= AÑO_MIN_VALIDO else None

    if isinstance(v, str):
        s = RE_HORA.sub("", v.strip())  # "1/7/2020  00:00:00" -> "1/7/2020"
        return None if s in ("", "0") else s

    if isinstance(v, (int, float)):
        return None if v == 0 else v

    return v


def encontrar_columna_dia(df):
    for col in df.columns:
        if str(col).strip().upper() == COL_DIA:
            return col
    return None


def procesar_hoja(df):
    """Devuelve (df_limpio, filas_eliminadas)."""
    # Tomo el día: 
    col = df.columns[0] 

    dias = df[col].map(limpiar_valor_dia)
    mask = dias.notna()
    out = df.loc[mask].copy()
    out[col] = dias[mask]
    return out.reset_index(drop=True), int((~mask).sum())


for archivo in archivos:
    print(f"\n=== {archivo.name} ===")
    hojas = pd.read_excel(archivo, sheet_name=None, engine="openpyxl")

    destino = OUT_DIR / archivo.name
    with pd.ExcelWriter(
        destino, engine="openpyxl", date_format="D/M/YYYY"
    ) as writer:
        for nombre, df in hojas.items():
            df_limpio, eliminadas = procesar_hoja(df)
            df_limpio.to_excel(writer, sheet_name=nombre, index=False)
            if eliminadas:
                print(f"  {nombre[:35]:<35} {eliminadas} filas eliminadas")
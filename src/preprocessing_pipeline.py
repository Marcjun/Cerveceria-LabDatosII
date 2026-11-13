from pathlib import Path
import warnings

import pandas as pd

from aux_functions_preprocess import (
    HOJAS_EXCLUIDAS, HOJAS_REQUERIDAS, ErrorEstructura,
    procesar_hoja, unir_hojas, validar_estructura,
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


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    dfs = cargar_excels(raw_dir.glob("Totalizadores Planta *.xlsx"))
    for nombre, df in dfs.items():
        print(f"{nombre}: {df.shape[0]} filas x {df.shape[1]} cols")
"""
auxiliar_functions.py -- Checksums SHA-256 de los datos del TP.

data/checksums.json guarda {ruta_relativa: sha256} de todos los archivos bajo
data/ (raw, processed y cualquier carpeta que se agregue). Se versiona en Git
y permite verificar que quien clona el repo tiene exactamente los mismos
.xlsx (que NO están en el repo).

Por qué SHA-256 y no MD5: MD5 tiene colisiones demostradas (se pueden fabricar
dos archivos distintos con el mismo hash), así que ya no garantiza integridad.

Uso
---
Desde Python:
    from auxiliar_functions import update_checksums, verify_checksums
    update_checksums()
    faltantes = verify_checksums()  # [{'path':..., 'sha256':...}, ...]

Desde la terminal (en la raíz del proyecto):
    python src/auxiliar_functions.py update   # recalcula y guarda checksums.json
    python src/auxiliar_functions.py verify   # lista faltantes; falla solo si hay archivos DISTINTOS
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Este archivo vive en src/, así que la raíz del proyecto es el padre del padre.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKSUMS_PATH = DATA_DIR / "checksums.json"

# Carpetas rastreadas. Por defecto es todo data/ (raw, processed y lo que se
# agregue a futuro), así ningún Excel puede quedar sin versionar.
# Para restringir: [DATA_DIR / "raw", DATA_DIR / "processed"]
TRACKED_DIRS = [DATA_DIR]

# No se hashean: el propio checksums.json (se referenciaría a sí mismo), los
# placeholder.txt (solo existen para que Git conserve carpetas vacías) y basura
# de SO / Excel (~$archivo.xlsx aparece al tener el Excel abierto).
_EXCLUDED_NAMES = {"placeholder.txt", ".gitkeep", ".DS_Store", "Thumbs.db"}


def _is_ignored(p: Path) -> bool:
    return (
        p == CHECKSUMS_PATH
        or p.name in _EXCLUDED_NAMES
        or p.name.startswith(("~$", ".~lock"))
    )


def hash_file(path, chunk=1 << 20):
    """SHA-256 de un archivo, leído en bloques de 1 MB (1 << 20 bytes) para
    no cargar los Excel enteros en memoria. Se abre en binario ("rb") para
    que funcione con cualquier tipo de archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# CSV que se comparan con tolerancia numérica en vez de byte a byte. En
# dataset_final.csv algunas features (p. ej. MES_sin) salen del sin/cos de
# numpy, que puede diferir en el último bit (~1e-16) entre Windows y Linux.
_TOLERANT_FILES = {"data/processed/dataset_final.csv"}
_TOLERANT_FLOAT_FORMAT = "%.12g"  # 12 cifras significativas


def hash_csv_tolerante(path):
    """SHA-256 de una versión canónica del CSV: se lee con pandas y se reescribe
    con los floats redondeados a 12 cifras significativas y saltos de línea LF.
    Absorbe diferencias de redondeo entre sistemas operativos, pero cualquier
    cambio real en los datos sigue cambiando el hash."""
    import pandas as pd  # import local: solo hace falta para estos archivos
    texto = pd.read_csv(path).to_csv(
        index=False, float_format=_TOLERANT_FLOAT_FORMAT, lineterminator="\n"
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def snapshot():
    """{ruta_relativa_a_la_raíz: sha256} de todos los archivos rastreados.

    Las rutas son relativas a la raíz y con '/' (as_posix), así el resultado
    es idéntico en cualquier máquina y sistema operativo.
    """
    snap = {}
    for folder in TRACKED_DIRS:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):  # rglob: recursivo
            if p.is_file() and not _is_ignored(p):
                rel = p.relative_to(ROOT).as_posix()
                snap[rel] = hash_csv_tolerante(p) if rel in _TOLERANT_FILES else hash_file(p)
    return dict(sorted(snap.items()))


def load_checksums():
    if not CHECKSUMS_PATH.exists():
        return {}
    text = CHECKSUMS_PATH.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}  # archivo vacío = primera vez


def save_checksums(snap):
    CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_PATH.write_text(
        json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _diff(expected, actual):
    """Compara dos snapshots -> (sin_registrar, distintos, faltantes)."""
    untracked = sorted(p for p in actual if p not in expected)
    missing = sorted(p for p in expected if p not in actual)
    modified = sorted(p for p in actual if p in expected and actual[p] != expected[p])
    return untracked, modified, missing


def update_checksums(confirm_removals=False):
    """Recalcula los hashes de data/ y guarda data/checksums.json.

    confirm_removals : si hay archivos registrados que ya no están en disco,
        hay que pasar True. Es una protección: alguien que clonó el repo sin
        los .xlsx NO debe pisar checksums.json declarándolos "borrados".
    """
    previous = load_checksums()
    current = snapshot()
    added, modified, removed = _diff(previous, current)

    if not (added or modified or removed):
        print("checksums: sin cambios.")
        return current

    print(f"checksums: {len(added)} nuevos, {len(modified)} modificados, "
          f"{len(removed)} borrados.")
    for label, paths in (("+", added), ("~", modified), ("-", removed)):
        for p in paths:
            print(f"  {label} {p}")

    if removed and not confirm_removals:
        raise RuntimeError(
            f"Hay archivos registrados que ya no están en disco: {removed}. "
            "Si no es un borrado intencional, falta copiar los datos (ver "
            "README). Si lo es, llamá a update_checksums(confirm_removals=True)."
        )

    save_checksums(current)
    print(f"checksums: guardado en {CHECKSUMS_PATH.relative_to(ROOT).as_posix()}")
    return current


def verify_checksums(strict=False, verbose=True):
    """Compara data/ contra checksums.json y devuelve la lista de archivos
    registrados que FALTAN en disco, cada uno con su checksum esperado:

        [{"path": "data/raw/x.xlsx", "sha256": "ab12..."}, ...]

    Lista vacía = no falta nada. Qué es error y qué no:
      - FALTA (registrado pero no está en disco): NO es error. Se devuelve en
        la lista (útil para saber qué hay que copiar y con qué hash debe quedar).
      - DISTINTO (está, pero el contenido no coincide): error -> RuntimeError.
      - SIN REGISTRAR (está en disco pero no en checksums.json, p. ej. archivos
        de otra rama): solo advertencia, salvo strict=True, que lo trata como error.
    """
    expected = load_checksums()
    if not expected:
        raise FileNotFoundError(
            f"No hay checksums en {CHECKSUMS_PATH.relative_to(ROOT).as_posix()}. "
            "Corré 'update' primero."
        )

    untracked, modified, missing = _diff(expected, snapshot())
    missing_list = [{"path": p, "sha256": expected[p]} for p in missing]

    if verbose:
        for m in missing_list:
            print(f"  FALTA          {m['path']}  (sha256 esperado: {m['sha256']})")
        for p in modified:
            print(f"  DISTINTO       {p}  (el contenido no coincide)")
        for p in untracked:
            print(f"  SIN REGISTRAR  {p}  (correr 'update' para versionarlo)")
        n_ok = len(expected) - len(missing) - len(modified)
        print(f"verify: {n_ok}/{len(expected)} archivos coinciden; "
              f"{len(missing)} faltan, {len(modified)} distintos, "
              f"{len(untracked)} sin registrar.")

    errors = []
    if modified:
        errors.append(f"{len(modified)} archivo(s) con contenido distinto al registrado")
    if strict and untracked:
        errors.append(f"{len(untracked)} archivo(s) sin registrar (strict=True)")
    if errors:
        raise RuntimeError("verify falló: " + "; ".join(errors) + ".")

    return missing_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checksums de los datos del TP")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_upd = sub.add_parser("update", help="recalcular y guardar checksums.json")
    p_upd.add_argument("--confirm-removals", action="store_true",
                       help="aceptar que hay archivos borrados")
    p_ver = sub.add_parser("verify", help="verificar data/ contra checksums.json")
    p_ver.add_argument("--strict", action="store_true",
                       help="tratar como error los archivos sin registrar")

    args = parser.parse_args()
    if args.cmd == "update":
        update_checksums(confirm_removals=args.confirm_removals)
    else:
        try:
            verify_checksums(strict=args.strict)  # faltantes: se listan, sin error
        except (RuntimeError, FileNotFoundError) as e:
            print(e)
            sys.exit(1)
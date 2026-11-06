rellenar

### 1 - Clonar el repositorio
```
git clone https://github.com/Marcjun/Cerveceria-LabDatosII
cd Cerveceria-LabDatosII
```

### 2 - Crear el enviroment
Asegurarse de tener conda instalado.

``` 
conda create -n cervecera_env python=3.12 -y
conda activate cervecera_env
pip install uv
uv pip install -r requirements.txt
```
Siempre tener cervecera_env activado. En caso que se desactive, correr:
``` 
conda activate cervecera_env
```

### 3 - Agregar los datos y verificar checksums iniciales

**Rellenar como poner los datos de los 3 totalizadores, y cambiarle el espacio en el - a uno**

```
python src/auxiliar_functions.py verify
```

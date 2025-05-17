import pandas as pd
import os

# Definir el directorio donde están los archivos Excel
dir = '../data/raw/precios_bolsa'  

# Crear una lista para almacenar los DataFrames
dfs = []


# Iterar sobre los archivos en la carpeta
for archivo in os.listdir(dir):
    if archivo.endswith('.xlsx') or archivo.endswith('.xls'):
        # Leer el archivo Excel y agregarlo a la lista
        df = pd.read_excel(os.path.join(dir, archivo))
        dfs.append(df)

# Concatenar todos los DataFrames en uno solo
df_combinado = pd.concat(dfs, ignore_index=True)

# Guardar el DataFrame combinado en un nuevo archivo Excel
df_combinado.to_excel('archivo_combinado.xlsx', index=False)
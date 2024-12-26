import os
import zipfile
from pathlib import Path    
import pandas as pd
import sys
import asyncio
import time
import urllib3

# Definimos la ruta de nuestros datos
RAW_DATA_PATH = Path('data/raw')

# Descomprime todos los archivos con extensión .zip en el directorio.
def unzip_data():
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    for file in RAW_DATA_PATH.glob('*.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_PATH)
            print(f"Extracted: {file}")

# Llamada a la API y guardado de los datos en un archivo .csv
def API_data_to_csv():
    # Importar funciones definidas para llamar a la API
    sys.path.append('src')
    from data.get_user_data_api import get_user_list, fetch_all_users_data

    # Llamada a la API para obtener el listado de user_id 
    users_list = get_user_list()
    print(f"{len(users_list)} usuarios encontrados.")

    # Llamadas API asincrónicas
    start_time = time.time()
    all_users_data = asyncio.run(fetch_all_users_data(users_list))
    end_time = time.time()
    print(f"Llamadas API resueltas en {end_time - start_time:.2f} segundos")

    # Crear DataFrame y llevar a .csv
    users_df = pd.DataFrame(all_users_data)
    users_df.to_csv(RAW_DATA_PATH / 'users_data.csv', index=False)
    print("DataFrame written to 'users_data.csv'")

# Lectura a pandas DataFrame.
def read_data():
    train_df = pd.read_csv(         RAW_DATA_PATH / 'train.csv')
    test_df = pd.read_csv(          RAW_DATA_PATH / 'test.csv')
    products_df = pd.read_pickle(   RAW_DATA_PATH / 'products.pkl')
    users_df = pd.read_csv(         RAW_DATA_PATH / 'users_data.csv')
    
    return train_df, test_df, products_df, users_df

if __name__ == "__main__":
    
    # Descomprimimos los archivos:
    # unzip_data()

    # Obtenemos los datos de la API y los guardamos en un archivo .csv
    # API_data_to_csv()

    # Leemos todos los .csv y .pkl
    train_df, test_df, products_df, users_df = read_data()

    print("\nTrain data:")
    print(train_df.head())
    print("\nTest data:")
    print(test_df.head())
    print("\nProducts data:")
    print(products_df.head())
    print("\nUsers data:")
    print(users_df.head())
    
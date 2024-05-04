import pandas as pd
import os

# Para todas las imagenes del folder data/images, guardo una lista que coja del nombre img_i_j.jpg, y guarde en cada posicion de la lista una tupla (i,j), siendo i la row del csv y j la columna del csv
image_tuples = []
for filename in os.listdir('./data/images'):
    # Elimina la extensión .jpg y divide el nombre del archivo por '_'
    parts = filename[:-4].split('_')
    # Extrae i y j del nombre del archivo
    i = int(parts[1])
    # Resta 1 a j para ajustar el índice de la columna
    j = int(parts[2]) - 1
    # Agrega la tupla (i, j) a la lista
    image_tuples.append((i, j))

df = pd.read_csv('./data/inditextech_hackupc_challenge_images.csv')
for i, j in image_tuples:
    # Verifica que los índices son válidos
    if i < len(df) and j < len(df.columns):
        # Accede al valor en la ubicación (i, j) del DataFrame
        value = df.iloc[i, j]
        print(value)
    else:
        print(f"Índices inválidos: {i}, {j}")


### Clusteritzem 

import urllib.parse

# Inicializa las listas
imagesSeason = []
imagesProductType = []
imagesSection = []
valid_image_tuples = []

for i, j in image_tuples:
    # Verifica que los índices son válidos
    if i < len(df) and j < len(df.columns):
        # Accede al valor en la ubicación (i, j) del DataFrame
        url = df.iloc[i, j]
        # Parsea la URL y divide el path por '/'
        parts = urllib.parse.urlparse(url).path.split('/')
        # Asegúrate de que el path tiene suficientes partes
        if len(parts) >= 5:
            # Si la temporada es 'public', salta al siguiente ciclo del bucle
            if parts[5] == 'public':
                continue
            # Agrega las partes relevantes a las listas
            imagesSeason.append(parts[5])
            imagesProductType.append(parts[6])
            imagesSection.append(parts[7])
            # Agrega la tupla a la lista de tuplas válidas
            valid_image_tuples.append((i, j))
        else:
            print(f"URL inválida: {url}")
    else:
        print(f"Índices inválidos: {i}, {j}")

# Imprime las clases existentes posibles de las tres listas
print(set(imagesSeason))
print(set(imagesProductType))
print(set(imagesSection))

# Imprime las tuplas válidas
#print(valid_image_tuples)
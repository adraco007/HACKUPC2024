# En este archivo preprocesamos los datos para que sean más fáciles de manejar y de entender.
from PIL import Image, ImageOps
import os
import numpy as np
import time

def resize_normalize_standardize_image(input_path, target_size=(224, 224)):
    """
    Carga una imagen desde una ruta, la redimensiona manteniendo la relación de aspecto,
    la normaliza y estandariza para modelos preentrenados en ImageNet,
    y devuelve como un array de NumPy.

    Parámetros:
        input_path (str): Ruta al archivo de imagen original.
        target_size (tuple): Tamaño deseado para la imagen como una tupla (ancho, alto).

    Retorna:
        numpy.ndarray: Imagen redimensionada, normalizada y estandarizada como un array de NumPy.
    """
    # Valores de media y desviación estándar para la normalización de ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with Image.open(input_path) as img:
        # Redimensiona la imagen manteniendo la relación de aspecto
        img = ImageOps.fit(img, target_size, Image.LANCZOS, 0, (0.5, 0.5))
        
        # Convertir la imagen a un array de NumPy y normalizar los píxeles al rango 0-1
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Estandarizar la imagen utilizando los valores de media y desviación estándar
        img_array = (img_array - mean) / std
        
        return img_array

def process_images(input_directory, output_directory, target_size=(224, 224)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Crear el directorio de salida si no existe

    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        # Asegurarse de que el archivo de salida tiene la extensión .npy
        base_filename = os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(output_directory, base_filename)

        if os.path.isfile(input_path):
            arr = resize_normalize_standardize_image(input_path, target_size)
            np.save(output_path, arr)  # Guarda el array en formato .npy
            #print(f"Processed and saved as NPY: {output_path}")

input_dir = './data/images'
output_dir = './data/processed_images'
t0 = time.time()
process_images(input_dir, output_dir)
t1 = time.time()
t_total = t1-t0
print(t_total)

# Vamos a leer un array de NumPy guardado para comprobar que se ha guardado correctamente
"""sample_image = np.load(os.path.join(output_dir, 'img_65_2.npy'))
print(sample_image)"""
import matplotlib.pyplot as plt
import numpy as np
import os

def show_image_from_array(image_array):
    """
    Función para mostrar una imagen representada por un array NumPy
    """
    # Reescala los valores del array al rango [0, 1] si no están en ese rango
    if np.min(image_array) < 0 or np.max(image_array) > 1:
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # Muestra la imagen utilizando Matplotlib
    plt.imshow(image_array)  # Muestra la imagen representada por el array NumPy
    plt.axis('off')  # Oculta los ejes de la imagen
    plt.show()  # Muestra la imagen en una ventana gráfica

output_dir = './data/processed_images'  # Directorio donde se guardan las imágenes procesadas
sample_image = np.load(os.path.join(output_dir, 'img_135_1.npy'))  # Carga un array NumPy desde un archivo .npy

# Llama a la función show_image_from_array para mostrar la imagen representada por el array NumPy cargado
show_image_from_array(sample_image)

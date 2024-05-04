# En este archivo preprocesamos los datos para que sean más fáciles de manejar y de entender.
from PIL import Image, ImageOps
import os

def resize_image(input_path, output_path, target_size=(224, 224)):
    with Image.open(input_path) as img:
        img = ImageOps.fit(img, target_size, Image.LANCZOS, 0, (0.5, 0.5))
        img.save(output_path)

def process_images(input_directory, output_directory, target_size=(224, 224)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Crear el directorio de salida si no existe

    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        if os.path.isfile(input_path):
            resize_image(input_path, output_path, target_size)
            print(f"Processed {filename}")

input_dir = './data/images'
output_dir = './data/processed_images'

process_images(input_dir, output_dir)
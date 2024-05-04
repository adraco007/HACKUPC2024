import clip
import torch
from PIL import Image
import os
import time
import numpy as np


class ClipModel():
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.embeddings_folder = './data/embeddings/'
        if not os.path.exists(self.embeddings_folder):
            os.makedirs(self.embeddings_folder)

    def _process_image(self, image_path):
        # Función para cargar y procesar una imagen
        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image

    def process_images(self, image_folder='./data/images/'):
        # Carpeta con tus imágenes de moda
        image_folder = './data/images/'
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

        # Verificar si la carpeta de embeddings existe, si no, crearla
        embeddings_folder = './data/embeddings/'
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

        start_time = time.time() # Timer
        # Generar y guardar embeddings para cada imagen
        for image_file in image_files:
            image = self._process_image(image_file)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            # Crear un nombre de archivo basado en el nombre de la imagen original
            base_filename = os.path.basename(image_file)
            embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
            embedding_filepath = os.path.join(embeddings_folder, embedding_filename)
            
            # Guardar el embedding en un archivo
            torch.save(image_features, embedding_filepath)
        end_time = time.time() # Timer
        print("Embeddings generated and saved for all images.")

        print(f"Time taken: {end_time - start_time} seconds")

    def process_selected_images(self, vector_images, images_path='./data/images/', embedding_path='./data/embeddings/'):
        # Carpeta con tus imágenes de moda
        image_files = sorted(os.listdir(images_path))

        # Pasar a numpy
        vector_images = np.array(vector_images)

        for idx in np.argwhere(vector_images == 1):
            print(idx[0])
            image_file = image_files[idx[0]]
            image_file = os.path.join(images_path, image_file)
            image = self._process_image(image_file)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Crear un nombre de archivo basado en el nombre de la imagen original
            base_filename = os.path.basename(image_file)
            embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
            embedding_filepath = os.path.join(embedding_path, embedding_filename)

            # Guardar el embedding en un archivo
            torch.save(image_features, embedding_filepath)
        

model = ClipModel()
#model.process_images()

start_time = time.time()
model.process_selected_images([0,0,0,0,0,1,1,1,0,0], images_path='./data/images/', embedding_path='./data/embeddings_test/')
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
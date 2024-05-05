import clip
import torch
from PIL import Image
import os
import time
import numpy as np
import pickle

class ClipModel():
    def __init__(self, download = False):
        self.device = "cpu"
        if download:
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
                image_features = image_features[0]
            # Crear un nombre de archivo basado en el nombre de la imagen original
            base_filename = os.path.basename(image_file)
            embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
            embedding_filepath = os.path.join(embeddings_folder, embedding_filename)
            
            # Guardar el embedding en un archivo
            torch.save(image_features, embedding_filepath)
        end_time = time.time() # Timer
        print("Embeddings generated and saved for all images.")

        print(f"Time taken: {end_time - start_time} seconds")

    def process_selected_images(self, vector_images, images_path='./data/images/', embedding_path='./data/embeddings_test/'):
        # Carpeta con tus imágenes de moda
        image_files = sorted(os.listdir(images_path))

        # Pasar a numpy
        vector_images = np.array(vector_images)

        # Lista de embeddings
        embeddings = {}
        indexes = {}

        for idx in np.argwhere(vector_images == 1):
  
            image_file = image_files[idx[0]]
            image_filepath = os.path.join(images_path, image_file)
            image = self._process_image(image_filepath)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Crear un nombre de archivo basado en el nombre de la imagen original
            base_filename = os.path.basename(image_filepath)
            embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
            embedding_filepath = os.path.join(embedding_path, embedding_filename)
        
            # Guardar el embedding en un archivo
            torch.save(image_features, embedding_filepath)
            embeddings[image_file] = (image_features, idx)
            embeddings[image_file] = image_features[0]
            indexes[image_file] = idx[0]
        print(image_features[0].shape)
        print(embeddings[image_file].shape)
        return embeddings, indexes

    def select_embeddings(self, vector_images):
        vector_images = np.array(vector_images)
        indexes = np.argwhere(vector_images==1)
        x = np.array(self.list_dirs_img)

        filenames = x[indexes]
        filenames = filenames.flatten()

        embeddings = {}
        indexes = {}
        for file_name in filenames:
            embeddings[file_name] = self.embeddings_dict[file_name]
            indexes[file_name] = self.embeddings_index_dict[file_name]
        
        return embeddings, indexes

    def load_embeddings(self, embeddings_folder='./data/embeddings/', image_folder='./data/images/'):
        # Cargar los embeddings de las imágenes
        embeddings = {}
        self.list_dirs_emb = os.listdir(embeddings_folder)
        self.list_dirs_emb.sort()
        
        self.list_dirs_img = os.listdir(image_folder)
        self.list_dirs_img.sort()
        
        embedding_dict = {}
        for index in range(len(self.list_dirs_emb)):
            embedding_dict[self.list_dirs_img[index]] = index
            embedding_filepath = os.path.join(embeddings_folder, self.list_dirs_emb[index])
            embedding = torch.load(embedding_filepath)
            embeddings[self.list_dirs_img[index]] = embedding

        self.embeddings_dict = embeddings
        self.embeddings_index_dict = embedding_dict
        return embeddings, embedding_dict
    
    def load_selection_embeddings(self, vector, embeddings_folder='./data/embeddings/'):
        filenames = os.listdir(embeddings_folder)
        filenames.sort()
        
        embeddings = {}

        vector_np = np.zeros(len(filenames))
        vector_np[:len(vector)] = np.array(vector)
        
        indexes = np.arange(0,len(filenames))
        
        filenames_np = np.array(filenames)
        filenames_np_select = filenames_np[vector_np == 1]
        
        print("pre",indexes)
        indexes_select = indexes[vector_np == 1]
        print("post", indexes_select)
        
        filenames_dict = {}
        for index in indexes_select:
            filenames_dict[filenames[index]] = index

        for filename in filenames_np_select:
            embedding_filepath = os.path.join(embeddings_folder, filename)
            embedding = torch.load(embedding_filepath)
            embeddings[filename] = embedding
        return embeddings, filenames_dict

    
    def process_select_image(self, image_path: str, embedding_path='./data/embeddings_test/'):

        image_filepath = image_path
        image = self._process_image(image_filepath)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Crear un nombre de archivo basado en el nombre de la imagen original
        base_filename = os.path.basename(image_filepath)
        embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
        embedding_filepath = os.path.join(embedding_path, embedding_filename)

        # Guardar el embedding en un archivo
        torch.save(image_features, embedding_filepath)
        embedding = image_features[0]

        return embedding
    
    def save_self(self, file_path='./models/clip_model.pkl'):
        try:
            # Guardar la instancia de la clase utilizando pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Modelo guardado como archivo binario en: {file_path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")

"""model = ClipModel()
model.save_self()"""

"""
start_time = time.time()
model.process_selected_images([0,0,0,0,0,1,1,1,0,0], images_path='./data/images/', embedding_path='./data/embeddings_test/')
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")"""
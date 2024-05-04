import time
import torch
import clip
import os
from PIL import Image
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class EmbedsVisualizer:
    def __init__(self, device="cpu"):
        self.device = device

    def visualize_nearest_images(self, embeddings: dict, selected_image_embedding, selected_image_filepath=None):

        num_pics = 5
        similarities = {}

        selected_image_embedding_processed = selected_image_embedding.squeeze().unsqueeze(0)  

        for filename, embedding in embeddings.items():
            embedding = embedding.squeeze().unsqueeze(0)  # Normaliza la forma para la comparación
            similarity = cosine_similarity(selected_image_embedding_processed, embedding)  # Calcula la similitud
            similarities[filename] = similarity.item()

        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_5_similar_images = sorted_similarities[:num_pics]

            # Suponiendo que ya has calculado las cinco imágenes más similares como en el código anterior
        
        top_5_similar_images = sorted_similarities[:num_pics]

        # Configurar la visualización
        fig, axes = plt.subplots(1, num_pics, figsize=(20, 4))  # Ajusta el tamaño según necesites
        fig.suptitle('Top 5 Similar Images')

        # Mostrar cada imagen
        for idx, (image_path, similarity) in enumerate(top_5_similar_images):
            image_path = os.path.join('./data/images', image_path.replace('.pt', '.jpg'))
            img = mpimg.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')  # Desactiva los ejes
            axes[idx].set_title(f"Sim: {similarity:.2f}")

        plt.show()

    def visualize_embeddings(self, embeddings: dict):
        num_pics = len(embeddings.keys())
        # Configurar la visualización
        fig, axes = plt.subplots(1, num_pics, figsize=(20, 4))  # Ajusta el tamaño según necesites
        fig.suptitle(f'Top {num_pics} Similar Images')

        # Lista con archivos del directorio /data/images
        image_files = os.listdir('./data/images')
        image_files.sort()

        # Mostrar cada imagen
        for idx, (image_file_name, (similarity, related_idx)) in enumerate(embeddings.items()):
            image_path = os.path.join('./data/images', f"{image_file_name}")
            img = mpimg.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')  # Desactiva los ejes
            axes[idx].set_title(f"Index: {image_file_name}")
            axes[idx].text(0.5, -0.1, f"Idx sim {image_files[related_idx][4:]}, Sim: {similarity:.2f}", size=10, ha='center', transform=axes[idx].transAxes)
        plt.show()

    def calculate_similarities(self, embedding_target_idx, similarities_matrix):
        # Calcular similitudes para un índice objetivo en la matriz de similitudes
        similarities = np.zeros((similarities_matrix.shape[0]))
        similarities[:embedding_target_idx] = similarities_matrix[embedding_target_idx, :embedding_target_idx]
        similarities[embedding_target_idx:] = similarities_matrix[embedding_target_idx:, embedding_target_idx]
        idx = np.argmax(similarities)
        return idx, similarities[idx]

    def get_max_similarity(self, embeddings: dict, num_images=5, images_path='./data/images/'):
        # Inicializar una matriz triangular para almacenar las similitudes
        num_embeddings = len(embeddings)
        similarity_matrix = np.zeros((num_embeddings, num_embeddings))

        # Obtener una lista ordenada de nombres de archivo
        filenames = sorted(embeddings.keys())

        # Procesar los embeddings y calcular las similitudes
        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):
                embedding_i = embeddings[filenames[i]].squeeze().unsqueeze(0)
                embedding_j = embeddings[filenames[j]].squeeze().unsqueeze(0)
                similarity = cosine_similarity(embedding_i, embedding_j)
                similarity_matrix[i, j] = similarity[0]  # Almacenar la similitud en la matriz triangular

        for i in range(num_embeddings):    
            similarity_matrix[i, i] = 0.0  # Establecer la diagonal en 0

        # Diccionario para almacenar las imágenes con su similitud correspondiente
        max_similarities = {}

        # Encontrar los primeros dos elementos con máxima similitud
        idx_max = np.argmax(similarity_matrix)
        row_idx, col_idx = np.unravel_index(idx_max, similarity_matrix.shape)
        max_similarities[filenames[row_idx]] = (similarity_matrix[row_idx, col_idx], col_idx)
        max_similarities[filenames[col_idx]] = (similarity_matrix[row_idx, col_idx], row_idx)

        # Actualizar la matriz de similitudes para excluir los elementos ya encontrados
        similarity_matrix[row_idx, col_idx] = 0
        similarity_matrix[col_idx, row_idx] = 0

        # Continuar encontrando los siguientes índices con máxima similitud iterativamente
        for _ in range(num_images - 1):
            # Encontrar el máximo en la matriz de similitudes entre pares de imágenes descubiertas y no descubiertas
            max_similarity = (-1, -1)
            for i, name_i in enumerate(max_similarities.keys()):
                for j in range(num_embeddings):
                    if filenames[j] not in max_similarities.keys():
                        similarity = similarity_matrix[filenames.index(name_i), j]
                        if similarity > max_similarity[0]:
                            max_similarity = (similarity, i)
                            next_idx = j
                            next_filename = filenames[j]
            # Agregar la imagen con máxima similitud al diccionario
            max_similarities[next_filename] = max_similarity

            # Actualizar la matriz de similitudes para excluir los elementos ya encontrados
            similarity_matrix[filenames.index(name_i), filenames.index(next_filename)] = 0
            similarity_matrix[filenames.index(next_filename), filenames.index(name_i)] = 0

        self.visualize_embeddings(max_similarities)
        return max_similarities




        
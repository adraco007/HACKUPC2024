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

        # Mostrar cada imagen
        for idx, (image_path, similarity) in range(num_pics):
            image_path = os.path.join('./data/images', image_path.replace('.pt', '.jpg'))
            img = mpimg.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')  # Desactiva los ejes
            axes[idx].set_title(f"Sim: {similarity:.2f}")

        plt.show()



    def get_max_similarity(self, embeddings: dict, num_images=5):
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
                similarity_matrix[i, j] = similarity[0, 0]  # Almacenar la similitud en la matriz triangular

        for i in range(num_embeddings):    
            similarity_matrix[i, i] = 0.0  # Establecer la diagonal en 0

        idx_max = np.argmax(similarity_matrix)
        row_idx, col_idx = np.unravel_index(idx_max, similarity_matrix.shape)

        self.visualize_embeddings({filenames[row_idx]: similarity_matrix[row_idx, col_idx], filenames[col_idx]: similarity_matrix[row_idx, col_idx]})
        return filenames[row_idx], filenames[col_idx], similarity_matrix[row_idx, col_idx]
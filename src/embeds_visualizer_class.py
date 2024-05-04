import time
import torch
import clip
import os
from PIL import Image
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


    def get_maxsimilarity(self, embeddings: dict, selected_image_embedding):
        similarities = {}

        selected_image_embedding_processed = selected_image_embedding.squeeze().unsqueeze(0)  

        for filename, embedding in embeddings.items():
            embedding = embedding.squeeze().unsqueeze(0)




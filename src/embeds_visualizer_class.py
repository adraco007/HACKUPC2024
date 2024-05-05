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
        for idx, (image_file_name, (similarity, realted_name)) in enumerate(embeddings.items()):
            image_path = os.path.join('./data/images', f"{image_file_name}")
            img = mpimg.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')  # Desactiva los ejes
            axes[idx].set_title(f"Index: {image_file_name}")
            axes[idx].text(0.5, -0.1, f"Idx sim {realted_name[4:]}, Sim: {similarity:.2f}", size=10, ha='center', transform=axes[idx].transAxes)
        plt.show()

    def calculate_similarities(self, embedding_target_idx, similarities_matrix):
        # Calcular similitudes para un índice objetivo en la matriz de similitudes
        similarities = np.zeros((similarities_matrix.shape[0]))
        similarities[:embedding_target_idx] = similarities_matrix[embedding_target_idx, :embedding_target_idx]
        similarities[embedding_target_idx:] = similarities_matrix[embedding_target_idx:, embedding_target_idx]
        idx = np.argmax(similarities)
        return idx, similarities[idx]

    def get_max_similarity(self, embeddings: dict, indexes:dict, vector_length: int, num_images=4, images_path='./data/images/'):
        t0=time.time()
        # Inicializar una matriz triangular para almacenar las similitudes
        assert(num_images >= 2)
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
        vector_indices = []
        similarity_list = np.zeros((num_images), dtype=np.float64)

        # Encontrar los primeros dos elementos con máxima similitud
        idx_max = np.argmax(similarity_matrix)
        row_idx, col_idx = np.unravel_index(idx_max, similarity_matrix.shape)
        max_similarities[filenames[row_idx]] = (similarity_matrix[row_idx, col_idx], filenames[col_idx])
        max_similarities[filenames[col_idx]] = (similarity_matrix[row_idx, col_idx], filenames[row_idx])

        vector_indices.append(indexes[filenames[row_idx]])
        vector_indices.append(indexes[filenames[col_idx]])

        return_similarity_array_index=0
        similarity_list[return_similarity_array_index] = similarity_matrix[row_idx, col_idx]
        return_similarity_array_index+=1
        similarity_list[return_similarity_array_index] = similarity_matrix[row_idx, col_idx]
        return_similarity_array_index+=1
        
        
        # Actualizar la matriz de similitudes para excluir los elementos ya encontrados
        similarity_matrix[row_idx, col_idx] = 0
        similarity_matrix[col_idx, row_idx] = 0

        

        # Continuar encontrando los siguientes índices con máxima similitud iterativamente
        for _ in range(num_images - 2):
            # Encontrar el máximo en la matriz de similitudes entre pares de imágenes descubiertas y no descubiertas
            max_similarity = (-1, -1)
            for i, name_i in enumerate(max_similarities.keys()):
                for j in range(num_embeddings):
                    if filenames[j] not in max_similarities.keys():
                        similarity = similarity_matrix[filenames.index(name_i), j]
                        if similarity > max_similarity[0]:
                            max_similarity = (similarity, name_i)
                            next_idx = j
                            next_filename = filenames[j]
            # Agregar la imagen con máxima similitud al diccionario
            max_similarities[next_filename] = max_similarity
            indx_vector_general = indexes[filenames[next_idx]]

            vector_indices.append(indx_vector_general)
            similarity_list[return_similarity_array_index] = max_similarity[0]
            return_similarity_array_index+=1
            # Actualizar la matriz de similitudes para excluir los elementos ya encontrados
            similarity_matrix[filenames.index(name_i), filenames.index(next_filename)] = 0
            similarity_matrix[filenames.index(next_filename), filenames.index(name_i)] = 0
        t1=time.time()
        print(f'Tiempo de ejecución: {t1-t0}')
        self.visualize_embeddings(max_similarities) # En caso de querer visualizar sin web
        
        print(vector_indices)
        return vector_indices, similarity_list


    def get_max_similarity_optimized(self, embeddings: dict, indexes: dict, num_images=4):
        t0=time.time()
        assert num_images >= 2
        
        num_embeddings = len(embeddings.keys())
        filenames = sorted(embeddings.keys())
        
        # Obtener todos los embeddings en una matriz
        embedding_matrix = np.array([embeddings[name].squeeze().unsqueeze(0)[0] for name in filenames])
        #print(embedding_matrix.shape)
        # Calcular todas las similitudes de una vez
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        #print(similarity_matrix.shape)
        # Configurar las similitudes entre pares de la misma imagen a cero
        np.fill_diagonal(similarity_matrix, 0)
        
        # Diccionario para almacenar las imágenes con su similitud correspondiente
        max_similarities = {}
        vector_indices = []
        indices_matriz = []
        similarity_list = np.zeros(num_images)
        related_names = []
        
        # Encontrar los primeros dos elementos con máxima similitud

        row_idx, col_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        indices_matriz.append(row_idx)
        indices_matriz.append(col_idx)
        max_similarities[filenames[row_idx]] = similarity_matrix[row_idx, col_idx]
        max_similarities[filenames[col_idx]] = similarity_matrix[row_idx, col_idx]
        
        #print(vector_indices)
        #print(filenames)
        #print(indexes)
        #print(row_idx)
        vector_indices.append(indexes[filenames[row_idx]])
        vector_indices.append(indexes[filenames[col_idx]])

        related_names.append(filenames[col_idx])
        related_names.append(filenames[row_idx])
        #print(f'indices_matriz: {indices_matriz}')
        #print(f'Similarity_matrix:\n{similarity_matrix}')

        similarity_list[:2] = similarity_matrix[row_idx, col_idx]
        
        # Configurar las similitudes a cero para los elementos ya encontrados
        similarity_matrix[row_idx, col_idx] = 0
        similarity_matrix[col_idx, row_idx] = 0

        # Matriz de similitudes temporal para encontrar los siguientes elementos
        temporal_similarity_matrix = similarity_matrix[indices_matriz,:]

        #print()
        #print(temporal_similarity_matrix)
        # Continuar encontrando los siguientes índices con máxima similitud iterativamente
        for i in range(2, num_images):
            
            
            # Encontrar el máximo en la matriz de similitudes
            fila, next_idx = np.unravel_index(np.argmax(temporal_similarity_matrix), temporal_similarity_matrix.shape)
            max_similarity = temporal_similarity_matrix[fila, next_idx]
            print(f'Se cumple?: {temporal_similarity_matrix[fila, next_idx]==similarity_matrix[next_idx, indices_matriz[fila]]}')
            # Agregar la imagen con máxima similitud al diccionario
            max_similarities[i] = max_similarity
            
            # Agregar los índices de vector
            vector_indices.append(embeddings[filenames[next_idx]])
            
            # Agregar la similitud a la lista
            similarity_list[i] = max_similarity
            related_names.append(filenames[indices_matriz[fila]])
            # Configurar las similitudes a cero para el nuevo elemento encontrado
            similarity_matrix[indices_matriz[fila], next_idx] = 0
            similarity_matrix[next_idx, indices_matriz[fila]] = 0

            indices_matriz.append(next_idx)
            temporal_similarity_matrix = similarity_matrix[indices_matriz,:]
            #print(f'similarity_matrix: \n{similarity_matrix}\n\n')
            #print(f'temporal_similarity_matrix: \n{temporal_similarity_matrix}\n\n')

        t1=time.time()
        print(f'Tiempo de ejecución: {t1-t0}')
        count=0
        viz_dict={}
        for idx in indices_matriz:
            if count==num_images:
                break
            viz_dict[filenames[idx]]= (similarity_list[count], related_names[count])
            count+=1
        self.visualize_embeddings(viz_dict)

        vector_indexes = [indexes[filenames[idx]] for idx in indices_matriz]
        print(indices_matriz)
        print(filenames[indices_matriz[3]])
        return vector_indexes, similarity_list


        
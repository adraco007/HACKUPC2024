import os
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree

# Suponiendo que usamos EfficientNet
model_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
model = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

embedding_model = tf.keras.Sequential([model])
embedding_model.compile()

def get_embeddings(images):
    return embedding_model.predict(images)

# Create a numpy array with all the images
# Define la ruta al directorio donde están almacenados tus archivos .npy
directory_path = './data/processed_images'

# Obtén una lista de nombres de archivos en el directorio
file_names = os.listdir(directory_path)

# Crea un array de NumPy a partir de los archivos .npy
images_paths = []
images = []
for file_name in file_names:
    images_paths = os.path.join(directory_path, file_name)
    file_path = os.path.join(directory_path, file_name)
    try:
        image = np.load(file_path)
        images.append(image)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")

# Convierte la lista de imágenes en un array de NumPy
images = np.array(images)
print(images)
embeddings = get_embeddings(images)
print(embeddings)
similarities = cosine_similarity(embeddings)
print('Similarities:')
print(similarities)
tree = KDTree(embeddings)
embedding_to_query = embeddings[33].reshape(1, -1)
distances, indices = tree.query(embedding_to_query, k=5)
print('Distances:')
print(distances)
print('Indices:')
print(indices)
#print number of similar photos
# Imprimir los paths de las imágenes más similares
print("Imagenes similares a:", images_paths[34])
for index in indices[0]:  # indices[0] porque la respuesta está en un array 2D
    print(images_paths[index])
#[[ 33 470 404 469 853]]
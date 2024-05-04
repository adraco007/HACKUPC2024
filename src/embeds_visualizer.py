import time



import torch
import clip
import os

device = "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

embeddings_dir = './data/embeddings/'

# Lista para almacenar los nombres de archivo de los embeddings
embedding_files = os.listdir(embeddings_dir)

# Diccionario para almacenar los embeddings
embeddings = {}
t0 = time.time()
# Cargar cada embedding desde su archivo respectivo
for file_name in embedding_files:
    embedding_path = os.path.join(embeddings_dir, file_name)
    embedding = torch.load(embedding_path, map_location=device)
    embeddings[file_name] = embedding
print(f'Embeddings loaded in t: {time.time()-t0}s')
t0 = time.time()
from PIL import Image

def preprocess_image(image_path, preprocess):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Añadir dimensión de batch
    return image

# Ruta a la imagen seleccionada
selected_image_path = './data/images/img_244_1.jpg'
selected_image = preprocess_image(selected_image_path, preprocess)

# Generar embedding para la imagen seleccionada
with torch.no_grad():
    selected_image_embedding = model.encode_image(selected_image)
    selected_image_embedding = selected_image_embedding.squeeze().unsqueeze(0)  # Normaliza la forma para la comparación


from torch.nn.functional import cosine_similarity

# Calcular la similitud del coseno entre la imagen seleccionada y todos los embeddings guardados
similarities = {}
for filename, embedding in embeddings.items():
    embedding = embedding.squeeze().unsqueeze(0)  # Normaliza la forma para la comparación
    similarity = cosine_similarity(selected_image_embedding, embedding)  # Calcula la similitud
    similarities[filename] = similarity.item()

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
top_5_similar_images = sorted_similarities[:5]

for image, similarity in top_5_similar_images:
    print(f"Imagen: {image}, Similitud: {similarity}")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Suponiendo que ya has calculado las cinco imágenes más similares como en el código anterior
num_pics = 5
top_5_similar_images = sorted_similarities[:num_pics]

# Configurar la visualización
fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Ajusta el tamaño según necesites
fig.suptitle('Top 5 Similar Images')

# Mostrar cada imagen
for idx, (image_path, similarity) in enumerate(top_5_similar_images):
    image_path = os.path.join('./data/images', image_path.replace('.pt', '.jpg'))
    img = mpimg.imread(image_path)
    axes[idx].imshow(img)
    axes[idx].axis('off')  # Desactiva los ejes
    axes[idx].set_title(f"Sim: {similarity:.2f}")

t1 = time.time()
print(f"{num_pics} images loaded in: {t1-t0} seconds")

plt.show()


import time

t0 = time.time()

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Cargar los embeddings desde el archivo
embeddings = torch.load('./data/image_embeddings.pt')

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
top_5_similar_images = sorted_similarities[:5]

# Configurar la visualización
fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Ajusta el tamaño según necesites
fig.suptitle('Top 5 Similar Images')

# Mostrar cada imagen
for idx, (image_path, similarity) in enumerate(top_5_similar_images):
    img = mpimg.imread(image_path)
    axes[idx].imshow(img)
    axes[idx].axis('off')  # Desactiva los ejes
    axes[idx].set_title(f"Sim: {similarity:.2f}")

t1 = time.time()
print(f"Time elapsed: {t1-t0} seconds")

plt.show()


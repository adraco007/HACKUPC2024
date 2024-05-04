import time
import torch
import clip
from PIL import Image
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import lru_cache

t0 = time.time()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

if torch.cuda.is_available():
    torch.cuda.set_device(device)  # Establece el dispositivo CUDA como global

# Cargar los embeddings desde el archivo
embeddings = torch.load('./data/image_embeddings.pt', map_location=device)

@lru_cache(maxsize=None)
def preprocess_image(image_path, preprocess, device):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)  # Asegurarse de añadir al dispositivo correcto aquí
    return image

# Ruta a la imagen seleccionada
selected_image_path = './data/images/img_111_1.jpg'
selected_image = preprocess_image(selected_image_path, preprocess, device)

# Generar embedding para la imagen seleccionada
with torch.no_grad():
    selected_image_embedding = model.encode_image(selected_image)
    selected_image_embedding = selected_image_embedding.squeeze().unsqueeze(0)  # Normaliza la forma para la comparación

embeddings_tensor = torch.stack(list(embeddings.values())).squeeze(1)
selected_image_embeddings_expanded = selected_image_embedding.expand_as(embeddings_tensor)

similarities_tensor = cosine_similarity(selected_image_embeddings_expanded, embeddings_tensor)

similarities = {filename: sim.item() for filename, sim in zip(embeddings.keys(), similarities_tensor)}

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
top_5_similar_images = sorted_similarities[:5]

for image, similarity in top_5_similar_images:
    print(f"Imagen: {image}, Similitud: {similarity}")

t1 = time.time()
print(f"Time elapsed: {t1-t0} seconds")

# Configurar la visualización
fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Ajusta el tamaño según necesites
fig.suptitle('Top 5 Similar Images')

# Mostrar cada imagen
for idx, (image_path, similarity) in enumerate(top_5_similar_images):
    img = mpimg.imread(image_path)
    axes[idx].imshow(img)
    axes[idx].axis('off')  # Desactiva los ejes
    axes[idx].set_title(f"Sim: {similarity:.2f}")



plt.show()


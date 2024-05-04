import clip
import torch
from PIL import Image
import os

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Función para cargar y procesar una imagen
def process_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image

# Carpeta con tus imágenes de moda
image_folder = './data/images/'
image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

# Diccionario para almacenar los embeddings
embeddings = {}

# Generar embeddings para cada imagen
for image_file in image_files:
    image = process_image(image_file)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    embeddings[image_file] = image_features

print("Embeddings generated for all images.")

# Guardar los embeddings en un archivo
torch.save(embeddings, './data/image_embeddings.pt')
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

# Verificar si la carpeta de embeddings existe, si no, crearla
embeddings_folder = './data/embeddings/'
if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

# Generar y guardar embeddings para cada imagen
for image_file in image_files:
    image = process_image(image_file)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Crear un nombre de archivo basado en el nombre de la imagen original
    base_filename = os.path.basename(image_file)
    embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
    embedding_filepath = os.path.join(embeddings_folder, embedding_filename)
    
    # Guardar el embedding en un archivo
    torch.save(image_features, embedding_filepath)

print("Embeddings generated and saved for all images.")
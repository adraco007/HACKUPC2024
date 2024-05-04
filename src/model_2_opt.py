import clip
import torch
from PIL import Image
import os
import time
import concurrent.futures

# Función para cargar y procesar una imagen
def process_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image

def train_on_batch(image_files):
    batch_images = [process_image(image_file) for image_file in image_files]
    batch_images = torch.cat(batch_images, dim=0)
    
    with torch.no_grad():
        image_features = model.encode_image(batch_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Procesar cada imagen en el batch
    for i, image_file in enumerate(image_files):
        # Crear un nombre de archivo basado en el nombre de la imagen original
        base_filename = os.path.basename(image_file)
        embedding_filename = os.path.splitext(base_filename)[0] + '.pt'
        embedding_filepath = os.path.join(embeddings_folder, embedding_filename)

        # Guardar el embedding en un archivo
        torch.save(image_features[i], embedding_filepath)

def chunks(lst, chunk_size):
    """Divide una lista en chunks de tamaño chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    # Cargar el modelo CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # Carpeta con tus imágenes de moda
    image_folder = './data/images/'
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

    # Verificar si la carpeta de embeddings existe, si no, crearla
    embeddings_folder = './data/embeddings/'
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    # Tamaño del batch
    batch_size = 164

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Dividir las imágenes en chunks de tamaño batch_size y entrenar en paralelo
        for image_chunk in chunks(image_files, batch_size):
            executor.submit(train_on_batch, image_chunk)

    end_time = time.time()

    print("Embeddings generated and saved for all images.")
    print(f"Time taken: {end_time - start_time} seconds")

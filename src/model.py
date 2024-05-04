import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np


class Auto_Encoder():
    def __init__(self):
        input_img = Input(shape=(224, 224, 3))  # Asumiendo imágenes RGB de tamaño 224x224

        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)  # Output es (28, 28, 8) si es necesario modificar

        # Bottleneck
        bottleneck = Flatten()(encoded)
        bottleneck = Dense(128, activation='relu', name='bottleneck')(bottleneck)  # Embeddings en el cuello de botella

        # Decoder
        x = Dense(np.prod(encoded.shape[1:]), activation='relu')(bottleneck)
        x = Reshape((encoded.shape[1], encoded.shape[2], encoded.shape[3]))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Asegurar que la salida tenga el mismo tamaño que la entrada

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')  # Usar MSE como función de pérdida

        autoencoder.summary()  # Ver la estructura del modelo

        self.autoencoder = autoencoder  # Guardar el modelo autoencoder

    def train_batch(self, noisy_image_batch, original_image_batch, num_batches, epochs=10, batch_size=32):
        """
        Entrena el autoencoder con batches de imágenes ruidosas y originales.

        Parámetros:
            noisy_image_batch (numpy.ndarray): Batch de imágenes con ruido.
            original_image_batch (numpy.ndarray): Batch de imágenes originales.
            epochs (int): Número de épocas de entrenamiento (default: 10).
            batch_size (int): Tamaño del batch (default: 32).
        """
        #num_batches = len(noisy_image_batch) // batch_size

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                noisy_batch = noisy_image_batch[start_idx:end_idx]
                original_batch = original_image_batch[start_idx:end_idx]

                # Entrenar con el batch actual
                batch_loss = self.autoencoder.train_on_batch(noisy_batch, original_batch)
                epoch_loss += batch_loss

                print(f"Batch {i + 1}/{num_batches} - Loss: {batch_loss:.4f}")

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / num_batches:.4f}")

    def denoise_image(self, noisy_image_batch):
        # Limpiar imágenes ruidosas utilizando el autoencoder entrenado
        denoised_images = self.autoencoder.predict(noisy_image_batch)
        return denoised_images




def create_image_batches(image_array, batch_size=32, target_size=(224, 224)):
    """
    Crea batches de imágenes a partir de un array de NumPy.

    Parámetros:
        image_array (numpy.ndarray): Array de NumPy que contiene las imágenes.
        batch_size (int): Tamaño del batch (default: 32).
        target_size (tuple): Tamaño deseado para las imágenes como una tupla (ancho, alto) (default: (224, 224)).

    Retorna:
        ImageDataGenerator: Generador de lotes de imágenes.
    """
    # Normalizar los valores de píxeles a [0, 1]
   
    image_array =  (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    # Crear un generador de imágenes con el array de imágenes
    datagen = ImageDataGenerator()

    # Generar batches de imágenes a partir del array
    image_batches = datagen.flow(image_array, batch_size=batch_size, target_size=target_size)

    return image_batches



import os
import numpy as np

def load_images_from_directory(directory, num_images=None):
    """
    Carga un número específico de imágenes del directorio especificado y las convierte en arrays de NumPy.

    Parámetros:
        directory (str): Directorio que contiene las imágenes.
        num_images (int): Número de imágenes a cargar (default: None, carga todas las imágenes).

    Retorna:
        numpy.ndarray: Array de NumPy que contiene las imágenes.
    """
    image_list = []
    image_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            image_array = np.load(filepath)
            image_list.append(image_array)
            image_count += 1
            if num_images is not None and image_count >= num_images:
                break
    image_array = np.array(image_list)
    return image_array



# Directorio que contiene las imágenes preprocesadas
processed_images_directory = './data/processed_images'

num_images_to_load = 32

# Cargar el número específico de imágenes del directorio y convertirlas en arrays de NumPy
image_array = load_images_from_directory(processed_images_directory, num_images=num_images_to_load)

# Tamaño del batch y tamaño deseado para las imágenes
batch_size = 32
target_size = (224, 224)

# Crear batches de imágenes a partir del array de NumPy
image_batches = create_image_batches(image_array, batch_size=batch_size, target_size=target_size)


# Directorio que contiene las imágenes preprocesadas
directory = './data/processed_images/'

image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

# Tamaño del batch y tamaño deseado para las imágenes
batch_size = 32
target_size = (224, 224)
num_batches = 1

# Crear batches de imágenes a partir del directorio especificado
image_batches = create_image_batches(directory, batch_size=batch_size, target_size=target_size, num_batches=num_batches)

next_batch = next(image_batches)

print(next_batch)
ae = Auto_Encoder()

ae.train_batch(next_batch, image_batches, num_batches=num_batches, epochs=10, batch_size=batch_size)
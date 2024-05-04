import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from model import base_model



image_batch = np.load('./data/processed_images/img_1_1.npy')

# Obtener los embeddings de las imágenes
image_embeddings = base_model.predict(image_batch)  # Suponiendo que tienes un lote de imágenes (image_batch)

# Reducir la dimensionalidad utilizando PCA
scaler = StandardScaler()
image_embeddings_scaled = scaler.fit_transform(image_embeddings)  # Escalar los embeddings antes de aplicar PCA
pca = PCA(n_components=2)  # 2 componentes principales para visualizar en un gráfico 2D
image_embeddings_pca = pca.fit_transform(image_embeddings_scaled)

# Visualizar las imágenes en el gráfico PCA
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(image_embeddings_pca):
    plt.scatter(x, y, c='b')  # Graficar cada punto en el espacio PCA
    plt.text(x, y, f"Image {i+1}", fontsize=8)  # Etiquetar cada punto con el número de imagen

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot of Image Embeddings')
plt.grid(True)
plt.show()
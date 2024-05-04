import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.load('./data/image_embeddings.pt', map_location=device)
embeddings_tensor = torch.stack(list(embeddings.values())).squeeze(1)
embeddings_np = embeddings_tensor.numpy() 
# Suponiendo que embeddings_np ya está definido y contiene tus embeddings como array de NumPy
pca = PCA()
pca.fit(embeddings_np)

# Calcular la varianza explicada acumulativa
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Preparar los datos para el gráfico
num_components = np.arange(len(cumulative_variance)) + 1  # Número de componentes (dimensiones)
zs = np.zeros_like(num_components)  # Eje ficticio (ahora será el eje X)

# Crear el gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Dibujar el gráfico, colocando el número de componentes en el eje Z
ax.plot(zs, cumulative_variance, num_components, marker='o', color='b')

# Añadir líneas verticales para cada punto para mejor visualización
for x, y, z in zip(zs, cumulative_variance, num_components):
    ax.plot([x, x], [y, y], [0, z], marker='_', color='red')

ax.set_xlabel('Eje Ficticio')
ax.set_ylabel('Varianza Acumulativa Explicada')
ax.set_zlabel('Número de Componentes')
ax.set_title('Explicabilidad por Número de Dimensiones en 3D')

plt.show()

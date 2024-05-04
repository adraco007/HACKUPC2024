import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.load('./data/image_embeddings.pt', map_location=device)
embeddings_tensor = torch.stack(list(embeddings.values())).squeeze(1)
embeddings_np = embeddings_tensor.numpy() 
pca = PCA()
pca.fit(embeddings_np)

# Varianza explicada
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Gráfico acumulativo de la varianza explicada
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance)
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulativa explicada')
plt.title('Varianza explicada por los componentes principales')
plt.grid(True)

# Determinar el número de componentes necesarios para explicar al menos el 80% de la varianza
target_variance = 0.8  # 80%
components_required = np.where(cumulative_variance >= target_variance)[0][0] + 1  # +1 porque el índice comienza en 0

# Añadir una línea horizontal y una línea vertical para marcar el 80% de explicabilidad
plt.axhline(y=target_variance, color='r', linestyle='--')
plt.axvline(x=components_required - 1, color='r', linestyle='--')  # -1 porque el índice comienza en 0
plt.annotate(f'80% explicabilidad con {components_required} componentes', 
             xy=(components_required, target_variance),
             xycoords='data', xytext=(-100, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

plt.show()

print(f"Se requieren {components_required} componentes para explicar al menos el 80% de la varianza.")
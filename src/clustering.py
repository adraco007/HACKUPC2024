from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch

# Asumiendo que embeddings_np es tu array NumPy de embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.load('./data/image_embeddings.pt', map_location=device)
embeddings_tensor = torch.stack(list(embeddings.values())).squeeze(1)
embeddings_np = embeddings_tensor.numpy() 
distortions = []
K = range(1, 10)  # Prueba algunos valores de k, por ejemplo de 1 a 10
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(embeddings_np)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Distorsión')
plt.title('El Método del Codo para Determinar k')
plt.show()

k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(embeddings_np)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(embeddings_np)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(ticks=range(k))
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.title('Visualización de Clusters de t-SNE')
plt.show()

from sklearn.metrics import silhouette_score

# Suponiendo que 'clusters' es el resultado de kmeans.fit_predict()
score = silhouette_score(embeddings_np, clusters)
print('Silhouette Score: %.3f' % score)

from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(embeddings_np, clusters)
print('Calinski-Harabasz Score: %.3f' % ch_score)

from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(embeddings_np, clusters)
print('Davies-Bouldin Score: %.3f' % db_score)

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# Asumiendo que 'embeddings_np' es tu array de embeddings como NumPy array
linked = linkage(embeddings_np, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendrograma de Clustering Jerárquico')
plt.show()

from sklearn.cluster import DBSCAN

# Asumiendo que 'embeddings_np' es tu array de embeddings como NumPy array
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps y min_samples son parámetros clave
clusters = dbscan.fit_predict(embeddings_np)
print(type(clusters))

plt.figure(figsize=(10, 7))
plt.scatter(embeddings_np[:, 0], embeddings_np[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Visualización de DBSCAN')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar()
plt.show()
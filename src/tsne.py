from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

# Suponiendo que embeddings_np es tu array de embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.load('./data/image_embeddings.pt', map_location=device)
embeddings_tensor = torch.stack(list(embeddings.values())).squeeze(1)
embeddings_np = embeddings_tensor.numpy() 
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(embeddings_np)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
plt.title('t-SNE de los Embeddings')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()

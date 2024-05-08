import numpy as np
from processor import Processor
import os

proc = Processor(download=True)
embeddings_dict = proc.embeddings_dict

keys_embs = sorted(os.listdir("./data/images"))
# Convert dictionary to a list of embeddings and a list of corresponding image paths
embeddings = np.array(list(embeddings_dict.values()))
image_paths = list(embeddings_dict.keys())
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path, zoom=0.08):
    return OffsetImage(plt.imread(path), zoom=zoom)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
x, y, z = reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2]
sc = ax.scatter(x, y, z)

# Adding image thumbnails
for x0, y0, z0, path in zip(x, y, z, image_paths):
    ab = AnnotationBbox(getImage(path), (x0, y0, z0), frameon=False, boxcoords="data")
    ax.add_artist(ab)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.show()
import os
import pandas as pd
import urllib.parse
from sklearn.preprocessing import LabelEncoder

class ImageClassifier:
    def __init__(self, image_folder='./data/images', csv_file='./data/inditextech_hackupc_challenge_images.csv'):
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.image_tuples = []
        self.df = pd.DataFrame()
        self.imagesSeason = []
        self.imagesProductType = []
        self.imagesSection = []
        self.encoder = LabelEncoder()
        self.cluster_indices = {}  # Dictionary to store indices for each cluster

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)
        for filename in os.listdir(self.image_folder):
            parts = filename[:-4].split('_')
            i = int(parts[1])
            j = int(parts[2]) - 1
            self.image_tuples.append((i, j))

    def classify_images(self):
        valid_image_tuples = []
        for i, j in self.image_tuples:
            if i < len(self.df) and j < len(self.df.columns):
                url = self.df.iloc[i, j]
                parts = urllib.parse.urlparse(url).path.split('/')
                if len(parts) >= 5:
                    if parts[5] == 'public':
                        continue
                    season, product_type, section = parts[5], parts[6], parts[7]
                    self.imagesSeason.append(season)
                    self.imagesProductType.append(product_type)
                    self.imagesSection.append(section)
                    valid_image_tuples.append((i, j))
                    # Create a cluster identifier
                    cluster_id = f"{season}_{product_type}_{section}"
                    # Append the image index to the cluster in the dictionary
                    if cluster_id not in self.cluster_indices:
                        self.cluster_indices[cluster_id] = []
                    self.cluster_indices[cluster_id].append(i)
                else:
                    print(f"Invalid URL: {url}")
            else:
                print(f"Invalid indices: {i}, {j}")
        self.image_tuples = valid_image_tuples

    def encode_classes(self):
        self.imagesSeason = self.encoder.fit_transform(self.imagesSeason)
        self.imagesProductType = self.encoder.fit_transform(self.imagesProductType)
        self.imagesSection = self.encoder.fit_transform(self.imagesSection)

    def print_cluster_indices(self):
        for cluster_id, indices in self.cluster_indices.items():
            print(f"Cluster {cluster_id}: {indices}")

    def run(self):
        self.load_data()
        self.classify_images()
        self.encode_classes()
        self.print_cluster_indices()
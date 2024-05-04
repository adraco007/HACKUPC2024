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
                    self.imagesSeason.append(parts[5])
                    self.imagesProductType.append(parts[6])
                    self.imagesSection.append(parts[7])
                    valid_image_tuples.append((i, j))
                else:
                    print(f"URL inválida: {url}")
            else:
                print(f"Índices inválidos: {i}, {j}")
        self.image_tuples = valid_image_tuples

    def encode_classes(self):
        self.imagesSeason = self.encoder.fit_transform(self.imagesSeason)
        self.imagesProductType = self.encoder.fit_transform(self.imagesProductType)
        self.imagesSection = self.encoder.fit_transform(self.imagesSection)

    def print_classes(self):
        print(f'imagesSeason: {set(self.imagesSeason)}')
        print(f'imagesProductType: {set(self.imagesProductType)}')
        print(f'imagesSection: {set(self.imagesSection)}')


    def create_intersection_list(self):
        intersectionList = []
        for season, productType, section in zip(self.imagesSeason, self.imagesProductType, self.imagesSection):
            intersectionList.append((season, productType, section))
        return intersectionList

    def encode_intersection_list(self, intersectionList):
        intersectionList = [str(i) for i in intersectionList]  # Convert tuples to string
        encodedIntersectionList = self.encoder.fit_transform(intersectionList)
        return encodedIntersectionList

    def run(self):
        self.load_data()
        self.classify_images()
        self.encode_classes()
        intersectionList = self.create_intersection_list()
        encodedIntersectionList = self.encode_intersection_list(intersectionList)
        self.print_classes()
        print(f'encodedIntersectionList: {set(encodedIntersectionList)}')

# Uso de la clase
classifier = ImageClassifier()
classifier.run()
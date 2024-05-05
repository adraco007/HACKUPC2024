#from PIL import Image
import numpy as np
import random
from src.model_clip import ClipModel
from src.embeds_visualizer_class import EmbedsVisualizer
from src.clusterPrevi import ImageClassifier
import pickle
import os
import time

"""
Communication between the GUI and the backend
"""

class Processor:
    def __init__(self, load_model=False, download=False, model_pathfile='./models/clip_model.pkl'):
        self.data = None
        if load_model:
            self.c = pickle.load(open(model_pathfile, 'rb'))
        else:
            self.c = ClipModel(download=download)
            if download:
                self.c.process_images()

        self.embeddings, self.index_dict = self.c.load_embeddings(embeddings_folder='./data/embeddings/')


<<<<<<< HEAD
    def find_outfit(self, selected_image_pathfile, vector, top_n=5):
        
        if vector is None:
=======
    def find_outfit(self, vector, top_n=6):
        selected_image_pathfile="./data/uploaded_images/image_01.jpg"
        c = ClipModel()
        if None in vector:
>>>>>>> 63053d795e002a13e50a27b27a65a2d31db34a04
            # Process the selected image to get its embedding
            embedding_selected_image = self.c.process_select_image(image_path=selected_image_pathfile, embedding_path='./data/embeddings/')
            # Load all embeddings
            #embeddings = c.load_embeddings(embeddings_folder='./data/embeddings/')

            # Remove the embedding of the selected image from the dictionary to avoid self-comparison
            selected_filename = os.path.basename(selected_image_pathfile)
            selected_embedding_key = os.path.splitext(selected_filename)[0] + '.pt'
            del self.embeddings[selected_embedding_key]

            # Compute cosine similarities
            similarities = {}
            for key, embedding in self.embeddings.items():
                sim = self.cosine_similarity(embedding_selected_image, embedding)
                similarities[key] = sim

            # Sort by similarity and select top N
            sorted_keys = sorted(similarities, key=similarities.get, reverse=True)[:top_n]

            # Convert filenames to numerical indices and return
<<<<<<< HEAD
            indices = []
            for key in sorted_keys:
                indices.append(self.index_dict)
            return indices
=======
            indices = [int(k.split('_')[1].split('.')[0]) for k in sorted_keys]
            return indices[1:]
>>>>>>> 63053d795e002a13e50a27b27a65a2d31db34a04
            

    def select_images(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """
        self.embeddings
        get_embeddings = False
        print("Processing images")

        print("Embeddings processed")
        vector_indices, vector_similaridades = EmbedsVisualizer().get_max_similarity(embeddings, indexes=indexes,  vector_length=len(vector))
        #image = Image.open('data/generated_images/image1.png')
        print(vector_similaridades)
        return vector_indices, vector_similaridades
    
    def select_images_optimized(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """
        embeddings, indexes = self.c.select_embeddings(vector)
        
        e = EmbedsVisualizer()
        vector_indices, vector_similaridades = e.get_max_similarity_optimized(embeddings=embeddings, indexes=indexes)
        #image = Image.open('data/generated_images/image1.png')
        """print(vector_similaridades)
        return vector_indices, vector_similaridades"""
        return vector_indices, vector_similaridades
    
    def get_embeddings(self, image_vector, image_path, selected_image_pathfile=None, created_model = False, model_pathfile = "./models/clip_model.pkl"):
        """
        Given a vector of images, return the embeddings of the images
        """
        if created_model:
            model = pickle.load(open(model_pathfile, 'rb'))
        else:
            try:
                model = ClipModel()
            except Exception as e:
                print(e)
                return None

        print("Model created")
        
        embeddings, indexes = model.process_selected_images(image_vector, images_path=image_path)
        if selected_image_pathfile:
            selected_image_embedding = model.process_select_image(image_path=selected_image_pathfile)
            return embeddings, selected_image_embedding
        else:
            return embeddings, indexes
        
    def load_embeddings(self, vector, created_model = False, model_pathfile = "./models/clip_model.pkl"):
        """
        Given a vector of images, return the embeddings of the images
        """
        if created_model:
            model = pickle.load(open(model_pathfile, 'rb'))
        else:
            try:
                model = ClipModel()
            except Exception as e:
                print(e)
                return None
        embeddings, indexes = model.load_selection_embeddings(vector)
        return embeddings, indexes

    def cosine_similarity(self, vec1, vec2):
        # Flatten the vectors to make sure they are 1-dimensional.
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        # Perform dot product only if they are properly aligned.
        if vec1.shape[0] == vec2.shape[0]:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        else:
            raise ValueError("Vector dimensions do not match.")
        
    def get_canta_vectors(self):
        """
        Get the vectors of the canta rero
        """
        ImageClassifier().load_data()
        pass    
"""

proc = Processor()
proc.c.load_embeddings(embeddings_folder='./data/embeddings/')

proc.select_images_optimized([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#Processor().find_outfit([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], selected_image_pathfile = "./data/images/img_0_1.jpg")

"""

"""
vector = [0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1]
model = ClipModel()
embeddings, indexes = model.get_embeddings()
embeddings2, indexes2 = model.load_embeddings(vector)

print(f'Embeddings: \nIndexes: {indexes}\n\n')
print(f'Embeddings2: \nIndexes2: {indexes2}')
"""
"""t0 = time.time()
p = Processor()
top_indices = p.find_outfit(vector=[None])
print("Indices of top similar images:", top_indices)
print("Time taken:", time.time()-t0)"""

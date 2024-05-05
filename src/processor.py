#from PIL import Image
import numpy as np
import random
from model_clip import ClipModel
from embeds_visualizer_class import EmbedsVisualizer
from clusterPrevi import ImageClassifier
import pickle

"""
Communication between the GUI and the backend
"""

class Processor:
    def __init__(self):
        self.data = None

    def find_outfit(self, selected_image_pathfile, vector, top_n=5):
        c = ClipModel()
        """
        Given an image, give back the x most similar images from the database
        """
        # Placeholder: 
        if vector == None:
            embedding_selected_image = c.process_select_image(image_path=selected_image_pathfile, embedding_path='./data/embeddings/')
            # Compute the simmularity with all the embeddings
            # First we load the embeddings
            embeddings = c.load_embeddings(embeddings_folder='./data/embeddings/')
            # Then we compute the similarity with the cosine similarity
            # We return the top 5 most similar images
            similarities = {}
    
            # Calcular la similaridad del coseno entre el vector dado y cada embedding en el diccionario
            for key, embedding in embeddings.items():
                sim = self.cosine_similarity(embedding_selected_image, embedding)
                similarities[key] = sim
            
            # Ordenar las similaridades y obtener las top_n claves
            sorted_keys = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
            print(sorted_keys)
            return sorted_keys
            

    def select_images(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """

        embeddings = self.get_embeddings(image_vector = vector, image_path="./data/images", created_model=True, model_pathfile="./models/clip_model.pkl")
        vector_indices, vector_similaridades = EmbedsVisualizer().get_max_similarity(embeddings,  vector_length=len(vector))
        #image = Image.open('data/generated_images/image1.png')
        print(vector_similaridades)
        return vector_indices, vector_similaridades
    
    def get_embeddings(self, image_vector, image_path, selected_image_pathfile=None, created_model = False, model_pathfile = "./models/clip_model.pkl"):
        """
        Given a vector of images, return the embeddings of the images
        """
        if created_model:
            model = pickle.load(open(model_pathfile, 'rb'))
        else:
            model = ClipModel()
        
        embeddings = model.process_selected_images(image_vector, images_path=image_path)
        if selected_image_pathfile:
            selected_image_embedding = model.process_select_image(image_path=selected_image_pathfile)
            return embeddings, selected_image_embedding
        else:
            return embeddings
    
    def cosine_similarity(self,vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
#Processor().select_images([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#Processor().find_outfit([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], selected_image_pathfile = "./data/images/img_0_1.jpg")

p = Processor()
p.find_outfit(selected_image_pathfile = "./data/uploaded/image_1.jpg", vector=None)


#from PIL import Image
import numpy as np
import random
from src.model_clip import ClipModel
from src.embeds_visualizer_class import EmbedsVisualizer
from src.clusterPrevi import ImageClassifier
import pickle

"""
Communication between the GUI and the backend
"""

class Processor:
    def __init__(self):
        self.data = None

    def find_outfit(self, vector):
        """
        Given an image, give back the x most similar images from the database
        """
        # Placeholder: 
        pass 

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
        
#Processor().select_images([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#Processor().find_outfit([0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], selected_image_pathfile = "./data/images/img_0_1.jpg")




#from PIL import Image
import numpy as np
import random
"""
Communication between the GUI and the backend

"""

class Processor:
    def __init__(self):
        self.data = None

    def find_outfit(self):
        """
        Given an image, give back the x most similar images from the database
        """
        # Get image from data/generated_images/image1.png
        #image = Image.open('data/generated_images/image1.png')
        
        # Placeholder: 
        pass 

    def select_images(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """

        # create a random vector of the same size as the vector given, fill it with 4 1s and the rest with 0s
        random_vector = np.zeros(len(vector))
        random_vector[:4] = 1
        random.shuffle(random_vector)
        return random_vector
        
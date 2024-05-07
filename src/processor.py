#from PIL import Image
import numpy as np
from src.model_clip import ClipModel
from src.embeds_visualizer_class import EmbedsVisualizer
from src.clusterPrevi import ImageClassifier
import pickle
import os
import shutil

"""
Communication between the GUI and the backend
"""

class Processor:
    """

    
    """
    def __init__(self, load_model=False, download=False, model_pathfile='./models/clip_model.pkl'):
        self.data = None
        self.model: ClipModel = None

        assert load_model or download, "A model has to be either loaded or downloaded"

        if load_model:
            self.model = pickle.load(open(model_pathfile, 'rb'))
        else:
            self.model = ClipModel(download=download)
            if download:
                self.model.process_images()

        self.embeddings, self.index_dict = self.model.load_embeddings(embeddings_folder='./data/embeddings/')


    def find_outfit(self, vector, top_n=5):
        """
        Given the image that the user has uploaded, return the top N most similar images
        """
        name = os.listdir('./data/uploaded_images/')[0]
        selected_image_pathfile = './data/uploaded_images/' + name

        if not self.model.downloaded:
            self.model.download(offline=True)
        if True:
            # Process the selected image to get its embedding
            embedding_selected_image = self.model.process_select_image(image_path=selected_image_pathfile, embedding_path='./data/embeddings/')
            # Load all embeddings


            # Compute cosine similarities
            similarities = {}
            print("aaaaaaaaaa",embedding_selected_image)
            for key, embedding in self.embeddings.items():
                sim = self.cosine_similarity(embedding_selected_image, embedding)
                similarities[key] = sim
            
            # Sort by similarity and select top N
            sorted_keys = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
            print("aaaaaaa", similarities["img_26_1.jpg"])
            print("aaaaaaa", sorted_keys)
            print("aaaaaaa", len(self.embeddings))
            # Convert filenames to numerical indices and return
            indices = []
            for key in sorted_keys:
                indices.append(self.index_dict[key])

            shutil.rmtree('./data/uploaded_images/')
            os.makedirs('./data/uploaded_images/')
            
            return indices
            

    def select_images(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """
        self.embeddings
        get_embeddings = False

        print("Processing images")

        vector_indices, vector_similaridades = EmbedsVisualizer().get_max_similarity(embeddings, indexes=indexes,  vector_length=len(vector))
        #image = Image.open('data/generated_images/image1.png')
        print("Images processed, returning the most similar ones")
        return vector_indices, vector_similaridades
    
    def select_images_optimized(self, vector):
        """
        given a vector of 1s and 0s, take from the images the ones that have a 1;
        separate them in clusters. inside each cluster, take the ones that are most similar from differents sets, take the cluster with the highest similarity, and return its top 4 similar images
        """

        print("Processing images")

        embeddings, indexes = self.model.select_embeddings(vector)
        
        e = EmbedsVisualizer()
        vector_indices, vector_similaridades = e.get_max_similarity_optimized(embeddings=embeddings, indexes=indexes)
        #image = Image.open('data/generated_images/image1.png')
        
        print("Images processed, returning the most similar ones")

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
        """
        Compute cosine similarity between two vectors.
        """
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
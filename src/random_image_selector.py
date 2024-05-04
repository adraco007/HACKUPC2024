"""
This class takes as the input an ammount of images, and then selects that ammount randomly from ./data/images. To do this, it checks how many images are in the folder, and then creates an array of zeros and ones, one being take the image, and zero not take it. Then it selects the images that have a one in the array.

Input: images to take
Output: array of zeros and ones, and total images in the folder
"""

import os
import numpy as np

class RandomImageSelector:
    def __init__(self, n_images):
        self.n_images = n_images
        self.images_names = []

    def select_images(self):
        self.images_names = os.listdir('./data/images')
        images_to_take = np.zeros(len(self.images_names))
        images_to_take[:self.n_images] = 1
        np.random.shuffle(images_to_take)
        return images_to_take, len(self.images_names)

selector = RandomImageSelector(10)
selector.select_images()
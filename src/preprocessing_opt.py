from PIL import Image, ImageOps
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

def resize_normalize_standardize_image(input_path, target_size=(224, 224)):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with Image.open(input_path) as img:
        img = ImageOps.fit(img, target_size, Image.LANCZOS, 0, (0.5, 0.5))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - mean) / std
        
        return img_array

def process_image(input_path, output_path, target_size=(224, 224)):
    arr = resize_normalize_standardize_image(input_path, target_size)
    np.save(output_path, arr)
    #print(f"Processed and saved as NPY: {output_path}")

def process_images_parallel(input_directory, output_directory, target_size=(224, 224)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with ThreadPoolExecutor() as executor:
        for filename in os.listdir(input_directory):
            input_path = os.path.join(input_directory, filename)
            base_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_directory, base_filename)

            if os.path.isfile(input_path):
                executor.submit(process_image, input_path, output_path, target_size)

input_dir = './data/images'
output_dir = './data/processed_images'

t0 = time.time()
process_images_parallel(input_dir, output_dir)
t1 = time.time()
t_total = t1-t0
print(t_total)
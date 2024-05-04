# First we read the csv file with the links of the images
import pandas as pd
import os
import requests
import re 

df = pd.read_csv('./data/inditextech_hackupc_challenge_images.csv')
num_pictures = 400

# Each row has 3 images, download first 10 rows
for i in range(num_pictures):
    try:
        path1 = df.iloc[i, 0]
        path2 = df.iloc[i, 1]
        path3 = df.iloc[i, 2]

        if pd.notnull(path1):
            img1 = requests.get(df.iloc[i, 0])
            if not re.match(r'.* have permission to access.*',str(img1.content)):
                with open(f'./data/images/img_{i}_1.jpg', 'wb') as f:
                    f.write(img1.content)
        if pd.notnull(path2):
            img2 = requests.get(df.iloc[i, 1])
            if not re.match(r'.* have permission to access.*',str(img2.content)):
                with open(f'./data/images/img_{i}_2.jpg', 'wb') as f:
                    f.write(img2.content)
        if pd.notnull(path3):
            img3 = requests.get(df.iloc[i, 2])
            if not re.match(r'.* have permission to access.*',str(img3.content)):
                with open(f'./data/images/img_{i}_3.jpg', 'wb') as f:
                    f.write(img3.content)

    except Exception as e:
        print(path1, path2, path3)
        print(e)
        print(f'Error downloading image {i}')
        continue



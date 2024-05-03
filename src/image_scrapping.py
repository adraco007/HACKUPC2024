# First we read the csv file with the links of the images
import pandas as pd
import os
import requests


df = pd.read_csv('./data/inditextech_hackupc_challenge_images.csv')
print(df.head())

# For each row in the dataframe we download the image, first 10 rows
for i in range(10*3):
    img_data = requests.get(df.iloc[i, 0]).content
    with open(f'./data/images/{i}.jpg', 'wb') as handler:
        handler.write(img_data)
"""
img_data = requests.get("https://static.zara.net/photos///2024/V/0/1/p/2910/009/051/2/w/2048/2910009051_3_1_1.jpg?ts=1709899261786").content
with open('image_name6.jpg', 'wb') as handler:
    handler.write(img_data)"""

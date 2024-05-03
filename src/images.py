# First we read the csv file with the links of the images
import pandas as pd
import os
df = pd.read_csv('./data/inditextech_hackupc_challenge_images.csv')
print(df.head())

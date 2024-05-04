import pandas as pd
import os
import requests
import re
from concurrent.futures import ThreadPoolExecutor

def download_image(url, filename):
    try:
        response = requests.get(url)
        #response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        if not re.match(r'.* have permission to access.*', str(response.content)):
            with open(filename, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images(df_row, index):
    for i, path in enumerate(df_row):
        if pd.notnull(path):
            download_image(path, f'./data/images/img_{index}_{i+1}.jpg')

def main():
    df = pd.read_csv('./data/inditextech_hackupc_challenge_images.csv')

    with ThreadPoolExecutor() as executor:
        for index, row in df.head(400).iterrows():
            executor.submit(download_images, row, index)


if __name__ == "__main__":
    if not os.path.exists('./data/images'):
        os.makedirs('./data/images')
    main()
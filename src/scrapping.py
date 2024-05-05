from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
import re
import pandas as pd
import os
import numpy as np


class ZaraScraper:
    def __init__(self, url):
        self.driver = webdriver.Chrome()
        self.url = url

    def scroll_page(self):
        self.driver.get(self.url)
        self.driver.maximize_window()

        height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            sleep(5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if height == new_height:
                break
            height = new_height

    def get_product_links(self, product_links):
        page_product_links = self.driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')
        for product in page_product_links:
            product_link = product.get_attribute('href')
            product_links.append(product_link)


list_of_links = [
    "https://www.zara.com/es/es/mujer-blazers-l1055.html?v1=2352684&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-vestidos-l1066.html?v1=2352823&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-faldas-l1299.html?v1=2353253&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-pantalones-shorts-l1355.html?v1=2353279&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-tops-l1322.html?v1=2353011&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-camisas-l1217.html?v1=2352910&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-camisetas-l1362.html?v1=2352955&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-jeans-l1119.html?v1=2353214&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-pantalones-l1335.html?v1=2353143&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-punto-l1152.html?v1=2352849&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-zapatos-l1251.html?v1=2353418&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-bolsos-l1024.html?v1=2353495&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-chaquetas-l1114.html?v1=2352724&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-punto-l1152.html?v1=2353051&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-prendas-exterior-l1184.html?v1=2352649&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-sudaderas-l1320.html?v1=2353089&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-prendas-exterior-chalecos-l1204.html?v1=2352738&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-beachwear-l1052.html?v1=2353512&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-accesorios-l1003.html?v1=2353548&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-ropa-interior-l4021.html?v1=2353568&regionGroupId=105",
    "https://www.zara.com/es/es/mujer-belleza-perfumes-l1415.html?v1=2353656&regionGroupId=105",
    "https://www.zara.com/es/es/origins-collection-l4661.html?v1=2351234&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-lino-l708.html?v1=2351649&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-camisas-l737.html?v1=2351464&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-camisetas-l855.html?v1=2351543&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-polos-l733.html?v1=2351616&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-pantalones-l838.html?v1=2351278&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-jeans-l659.html?v1=2351397&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-bermudas-l592.html?v1=2351786&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-beachwear-l590.html?v1=2378240&regionGroupId=105",
    "https://www.zara.com/es/es/man-crochet-l6272.html?v1=2351800&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-prendas-exterior-l715.html?v1=2378740&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-sudaderas-l821.html?v1=2351429&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-punto-l681.html?v1=2351499&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-sobrecamisas-l3174.html?v1=2351642&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-blazers-l608.html?v1=2351609&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-pantalones-cargo-l1780.html?v1=2351761&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-zapatos-zapatillas-l797.html?v1=2389259&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-zapatos-l769.html?v1=2352273&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-bolsos-l563.html?v1=2352310&regionGroupId=105",
    "https://www.zara.com/es/es/hombre-accesorios-perfumes-l551.html?v1=2352350&regionGroupId=105",
    "https://www.zara.com/es/es/kids-babygirl-collection-l5415.html?v1=2397212&regionGroupId=105",
    "https://www.zara.com/es/es/kids-girl-collection-l7289.html?v1=2396729&regionGroupId=105",
    "https://www.zara.com/es/es/kids-babyboy-collection-l5414.html?v1=2397220&regionGroupId=105",
    "https://www.zara.com/es/es/kids-boy-collection-l5413.html?v1=2397714&regionGroupId=105",
    "https://www.zara.com/es/es/ninos-recien-nacido-l474.html?v1=2403733&regionGroupId=105",
    "https://www.zara.com/es/es/kids-mini-view-all-l6750.html?v1=2404242&regionGroupId=105"
]

def save_list_to_csv(data_list, csv_file):
    df = pd.DataFrame(data_list, columns=['data'])
    df.to_csv(csv_file, index=False)

'''
# Uso de la clase
product_links = []
product_ids = []
for link in list_of_links:
    scraper = ZaraScraper(link)
    scraper.scroll_page()
    scraper.get_product_links(product_links)


# Uso de la función
save_list_to_csv(product_links, './data/product_links.csv')
'''



class LinksPhotoToProduct:
    def __init__(self):
        self.df = pd.DataFrame(columns=['photo_link', 'product_link'])
        self.image_links = self.extract_image_links_from_csv()
        self.product_links = list(set(pd.read_csv('./data/product_links.csv')['data'].tolist()))
        self.product_dict = {link.split('-p0')[1].split('.html')[0]: link for link in self.product_links if '-p0' in link}

    def extract_image_links_from_csv(self, csv_file='./data/inditextech_hackupc_challenge_images.csv'):
        df = pd.read_csv(csv_file)
        image_links = []
        for filename in os.listdir('./data/images'):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Asegúrate de que es una imagen
                parts = filename[:-4].split('_')
                i = int(parts[1])
                j = int(parts[2]) - 1
                if i < len(df) and j < len(df.columns):
                    url = df.iloc[i, j]
                    image_links.append(url)
        return image_links

    def link_photo_product(self):
        for photo in self.image_links:
            #print('Processing photo')
            parts = photo.split('/')
            result = parts[11] + parts[12]
            #print(result)

            # Buscar el product_link correspondiente en el diccionario
            product_link = self.product_dict.get(result, np.nan)
            print(f'Product id: {result}, Product link: {product_link}')

            # Agregar una nueva fila al DataFrame
            self.df = self.df._append({'photo_link': str(photo), 'product_link': str(product_link)}, ignore_index=True)

    def save_to_csv(self, csv_file='./data/links_photo_to_product.csv'):
        self.df.to_csv(csv_file, index=False)

link_processor = LinksPhotoToProduct()
link_processor.link_photo_product()
link_processor.save_to_csv()
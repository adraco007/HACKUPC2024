from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
import re

class ZaraScraper:
    def __init__(self, url):
        self.driver = webdriver.Chrome()
        self.url = url
        self.product_links = []
        self.product_ids = []

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

    def get_product_links(self):
        page_product_links = self.driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')
        for product in page_product_links:
            product_link = product.get_attribute('href')
            self.product_links.append(product_link)

    def get_product_ids(self):
        for product in self.product_links:
            match = re.search(r'p0(\d+)\.html', product)
            if match:
                product_id = match.group(1)
                self.product_ids.append(product_id)

    def find_product(self, photo):
        parts = photo.split('/')
        result = parts[11] + parts[12]

        for product_id in self.product_ids:
            if result == product_id:
                link = self.product_links[self.product_ids.index(product_id)]
                print(link)

# Usage
scraper = ZaraScraper('https://www.zara.com/es/es/hombre-camisetas-l855.html?v1=2351543&regionGroupId=105')
scraper.scroll_page()
scraper.get_product_links()
scraper.get_product_ids()
scraper.find_product('https://static.zara.net/photos///2024/V/0/2/p/0679/416/251/2/w/2048/0679416251_6_1_1.jpg?ts=1714473878015')
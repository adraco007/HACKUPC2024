# Importing the required libraries
from selenium import webdriver
from time import sleep
from csv import writer
from selenium.webdriver.common.by import By

# Specify the full path to the ChromeDriver executable
#chrome_driver_path = r"C:\Users\Dell\Downloads\chromedriver_win32\chromedriver.exe"
driver = webdriver.Chrome()

driver.get('https://www.zara.com/es/es/origins-collection-l4661.html?v1=2351234&regionGroupId=105&page=9')
driver.maximize_window()

# Scrolling the web page
height = driver.execute_script("return document.body.scrollHeight")
while True:
   driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
   sleep(5)
   new_height = driver.execute_script("return document.body.scrollHeight")
   if height == new_height:
        break
   height = new_height


product_links = []

# Getting the product elements
page_product_links = driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')

# Getting the product links
for product in page_product_links:
   product_link = product.get_attribute('href')
   product_links.append(product_link)

#print(product_links)

import re

product_ids = []

# Getting the product links
for product in page_product_links:
   product_link = product.get_attribute('href')
   
   # Use regex to find the product id
   match = re.search(r'p0(\d+)\.html', product_link)
   if match:
       product_id = match.group(1)
       product_ids.append(product_id)

print(product_ids)


photo = 'https://static.zara.net/photos///2024/V/0/3/p/5767/521/712/2/w/2048/5767521712_6_1_1.jpg?ts=1707751045954'

# Split the string by '/'
parts = photo.split('/')

# Take the 9th and 10th parts and concatenate them
result = parts[11] + parts[12]

print(result)
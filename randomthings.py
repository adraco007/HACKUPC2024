import os

names = os.listdir('data/images')
names.sort()
name = names[1]

print(name)

# format: img_x_y.jpg -> (x, y)
x = int(name.split('_')[1])
y = int(name.split('_')[2].split('.')[0])

print(x, y)

with open('data/inditextech_hackupc_challenge_images.csv', 'r') as f:
    lines = f.readlines()
    link = lines[x+1].split(',')[y-1][1:-1] # Header is the first line, every set has images 1-3
    print(link)

product_link = None

# Check if link exists in extraData\links_photo_to_product.csv
with open('extraData/links_photo_to_product.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if link in line:
            product_link = line.split(',')[1]
            print(f"Link: {link}")
            break
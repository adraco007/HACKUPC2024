from flask import Flask, render_template
from src.random_image_selector import RandomImageSelector
from src.processor import Processor
from flask import request, jsonify, send_from_directory
import os, time

# Look for a model in the data folder
# Get the names
names = os.listdir('models')
if names == []:
    print("No model found, downloading one and precomputing the embeddings, this may take a while")
    processor = Processor(download=True, load_model=False)
else:
    print("Model found, loading it")
    processor = Processor(download=False, load_model=True)
    print("Model loaded, launching the server")
    
app = Flask(__name__)

# Get templates
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/order')
def order():
    return render_template('order.html')

@app.route('/search')
def search():
    return render_template('search.html')

# Get images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory('data/generated_images', filename)

@app.route('/static/images/<path:filename>')
def serve_blank_image(filename):
    return send_from_directory('static/images', filename)
                               
""" Get the image from the index, alphanumerically sorted"""
@app.route('/images_from_index')
def images_from_index():
    try:
        index = int(request.args.get('index'))
    except:
        print("Error: unable to parse index")
        index = 0
    
    images = os.listdir('data/images')
    images.sort()
    image = images[index]
    print(f"Image selected: {image}")

    return send_from_directory('data/images', image)

# Functions
""" From the index of an image, get the link of the product in the zara website """
@app.route('/get_shop_link')
def get_shop_link():
    time.sleep(0.20)
    try:
        index = int(request.args.get('index'))
        print("Index parsed correctly")
    except:
        print("Error: index is not an integer")
        index = 0

    names = os.listdir('data/images')
    names.sort()
    name = names[index]

    # format: img_x_y.jpg -> (x, y)
    x = int(name.split('_')[1])
    y = int(name.split('_')[2].split('.')[0])

    with open('data/inditextech_hackupc_challenge_images.csv', 'r') as f:
        lines = f.readlines()
        link = lines[x+1].split(',')[y-1] # Header is the first line, every set has images 1-3

    product_link = None

    # Check if link exists in extraData\links_photo_to_product.csv
    with open('extraData/links_photo_to_product.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if link in line:
                product_link = line.split(',')[1]
                print(f"Link: {link}")
                break

    return jsonify({'url': product_link})

""" 
Given the number of images, select that random ammount of images from the ones
that are in the data/images folder. Then, get the 4 most similar images amongst them
"""
@app.route('/select_similar_images')
def select_similar_images():
    try:
        n_images = int(request.args.get('n_images'))
    except:
        print("Error: unable to parse n_images")
        n_images = 10
        
    selector = RandomImageSelector(n_images)
    images_to_take, total_images = selector.select_images()

    index_list, similarity_list = processor.select_images_optimized(images_to_take)
    similarity_list = similarity_list.tolist()
    similarity_list = [float(i) for i in similarity_list]
    index_list = [int(i) for i in index_list]

    # Return JSON response
    return jsonify({
        'indexs': index_list,
        'similarities': similarity_list,
        'total_images': total_images
    })

"""
Get the indexs of the most similar images to the uploaded image, 
given the settings that the user has selected
"""
@app.route('/search_outfit', methods=['POST'])
def process_uploaded_file():

    # Get the information from the form about the type of product we are looking for
    season = request.form.get('season') 
    product_type = request.form.get('style')
    section = request.form.get('section')
    if season == "Season":
        season = None
    if product_type == "Product Type":
        product_type = None
    if section == "Section":
        section = None

    # Get the uploaded image
    uploaded_file = request.files['file']
    uploaded_file.save('data/uploaded_images/' + uploaded_file.filename)

    indexes = processor.find_outfit([season, product_type, section])

    return jsonify({'indexes': indexes})

if __name__ == '__main__':
    app.run(debug=True)

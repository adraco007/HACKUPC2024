from flask import Flask, render_template
from src.random_image_selector import RandomImageSelector
from src.processor import Processor
from src.model_clip import ClipModel
from flask import request
from flask import jsonify
from flask import send_from_directory
import os

processor = Processor()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_uploaded_file', methods=['POST'])
def process_uploaded_file():
    # Get the formData from the request
    season = request.form.get('season')
    product_type = request.form.get('style')
    section = request.form.get('section')

    # Get the uploaded file
    uploaded_file = request.files['file']

    # Save the file
    uploaded_file.save('data/uploaded_images/' + uploaded_file.filename)

    if season == "Season":
        season = None
    if product_type == "Product Type":
        product_type = None
    if section == "Section":
        section = None

    indexs = processor.find_outfit([season, product_type, section])
    print(indexs)
    indexs.sort()

    # Return JSON response
    return jsonify({
        'indexs': indexs
    })



@app.route('/generate_image')
def generate_image():
    # Get parameters from request
    prompt = request.args.get('prompt')

    # Generate image
    print(f"Generating image with prompt: {prompt}")
    

    # No response needed, ok code only
    return '', 200

@app.route('/select_images')
def select_images():
    try:
        n_images = int(request.args.get('n_images'))  # Convert to int
        print("Number parsed correctly")
    except:
        print("Error: n_images is not an integer")
        n_images = 10  # Default value
        
    selector = RandomImageSelector(n_images)
    images_to_take, total_images = selector.select_images()

    index_list, similarity_list = processor.select_images_optimized(images_to_take)

    similarity_list = similarity_list.tolist()
    
    # map similarity to float 
    similarity_list = [float(i) for i in similarity_list]

    # Map index to int 
    index_list = [int(i) for i in index_list]
    print(index_list)
    print(index_list)
    print(similarity_list)
    # Return JSON response
    return jsonify({
        'indexs': index_list,
        'similarities': similarity_list
    })

@app.route('/images_from_index')
def images_from_index():
    try:
        index = int(request.args.get('index'))  # Convert to int
        print("Index parsed correctly: ", index)
    except:
        print("Error: index is not an integer")
        index = 0
    
    # From data/images take the image with the index
    # Get the names of all the images
    images = os.listdir('data/images')
    images.sort()
    image = images[index]
    print(f"Image selected: {image}")

    # Return the image
    return send_from_directory('data/images', image)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory('data/generated_images', filename)

# Route to the satic/blank.png image
@app.route('/static/blank.png')
def serve_blank_image():
    return send_from_directory('static', 'blank.png')

@app.route('/outfit')
def outfit():
    return render_template('outfit.html')

@app.route('/order')
def order():
    return render_template('order.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('get_image_link')
def get_image_link():
    try:
        index = int(request.args.get('index'))  # Convert to int
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

    # Check in data\inditextech_hackupc_challenge_images.csv for the link in that position
    with open('data/inditextech_hackupc_challenge_images.csv', 'r') as f:
        lines = f.readlines()
        link = lines[x+1].split(',')[y-1]

    product_link = None

    # Check if link exists in extraData\links_photo_to_product.csv
    with open('data/extraData/links_photo_to_product.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if link in line:
                product_link = line.split(',')[1]
                print(f"Link: {link}")
                break

    return jsonify({
        'url': product_link
    })


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template
from src.random_image_selector import RandomImageSelector
from flask import request
from flask import jsonify
from flask import send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_uploaded_file', methods=['POST'])
def process_uploaded_file():
    # Get the uploaded file
    uploaded_file = request.files['file']

    # Save the file
    uploaded_file.save('data/uploaded_images/' + uploaded_file.filename)

    # No response needed, ok code only
    return '', 200

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

    print(f"Images to take: {n_images}")

    # Return JSON response
    return jsonify({
        'imagesToTake': images_to_take.tolist(),  # Convert numpy array to list
        'totalImages': total_images
    })

@app.route('/get_images')
def get_images():
    pass # Cridar la funció que retorni el nom de les fotos que s'han de carregar

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


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template
from src.random_image_selector import RandomImageSelector
from flask import request
from flask import jsonify
from flask import send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)

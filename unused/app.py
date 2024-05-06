@app.route('/generate_image')

@app.route('/outfit')
def outfit():
    return render_template('outfit.html')

def generate_image():
    # Get parameters from request
    prompt = request.args.get('prompt')

    # Generate image
    print(f"Generating image with prompt: {prompt}")
    
    # No response needed, ok code only
    return '', 200
import os
import openai
import sys
import argparse
import requests
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import io
from PIL import Image


# Parameters
GPT_model = ""

# Key setup
def setup_openai(model = "gpt-3.5-turbo"):
    global GPT_model
    GPT_model = model
    # Setup OpenAI API
    local_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_key = os.path.join(local_dir, 'openai_key.txt')

    try:
        with open(path_to_key, 'r') as file:
            openai_key = file.read()
    except Exception as e:
            print(e)

    openai.api_key = openai_key
    os.environ['OPENAI_API_KEY'] = openai_key


    return openai_key

def image_generator(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_response_image(prompt, quality)
    return response

def generate_response_image(prompt, quality='high'):
    # Generate reference image from prompt
    my_key = setup_openai()

    if quality == 'low':
        size = "256x256"
    elif quality == 'medium':
        size = "512x512"
    elif quality == 'high':
        size = "1024x1024"

    # elif
    # we don't have that much money

    # Generate a nice image prompt from the text/keywords
    """messag=[{"role": "system", "content": "You are a keywords to image outfit prompt generator. From the given keywords or message, you will generate a prompt for Dalle to generate a nice garment that fits the description with white background, like a stock image for a fashion website."}]
    history_user = ["i'll give you some key words or a message. with all of them, you will generate a nice stock image of a garment that fits the description with white background."]
    history_bot = ["Yes, I'm ready! Please provide the keywords or message."]
    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})
    response = openai.chat.completions.create(
        model=GPT_model,
        messages=messag,
        max_tokens=200,
        temperature=0.8,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content"""

    # Generate image from prompt
    response = openai.images.generate(
    prompt="The following garment with white background for an online cloth store stock image: " + prompt + ", realist, high quality, fits fully in the image",
    n=1,
    size=size
    )
    image_url = response.data[0].url

    # Get the image from the url
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image content to BMP format
        image = Image.open(io.BytesIO(response.content))
    else:
        image = None

    # Get local path
    local_dir = os.path.dirname(os.path.abspath(__file__))
    # go one up to the parent folder
    local_dir = os.path.dirname(local_dir)
    
    image.save(os.path.join(local_dir, 'data', 'generated_images', 'image1.png'))

    return image
import requests

from flask import Flask,request

from io import BytesIO
import base64

from PIL import Image

import numpy as np

import sys
import os

app = Flask(__name__)

@app.route("/", methods = ['POST'])

def index():

    print('Request-form',list(request.form.keys()),file=sys.stderr)

    image_string = request.form['image']

    subscription_key = "d47074371b6f4084864d641b5677e03e"
    
    vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

    analyze_url = vision_base_url + "analyze"

    image_data = BytesIO(base64.b64decode(image_string))

    headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                  'Content-Type': 'application/octet-stream'}
    
    params     = {'visualFeatures': 'Description'}

    api_response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    
    api_response.raise_for_status()

    analysis = api_response.json()
    
    description = "startseq " + analysis["description"]["captions"][0]["text"].capitalize() + " endseq"
    
    return description

if(__name__ == "__main__"):
    app.run(host = '0.0.0.0',port = 5001)

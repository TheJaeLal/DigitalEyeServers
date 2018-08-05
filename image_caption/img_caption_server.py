from pickle import load

from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from flask import Flask,request

from io import BytesIO
import base64

#from scipy.misc import imresize

from PIL import Image
import cv2

import numpy as np

import sys
import os

# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    cnn = VGG16()
    # re-structure the model
    cnn.layers.pop()

    cnn = Model(inputs=cnn.inputs, outputs=cnn.layers[-1].output)
    
    # global graph
    # graph = tf.get_defualt_graph()

    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    
    # with graph.as_default():
    feature = cnn.predict(image, verbose=0)
    
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


app = Flask(__name__)

@app.route("/", methods = ['POST'])

def index():

    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 30
    # load the model
    model = load_model('model-ep004-loss3.634-val_loss3.981.h5')

    print('Request-form',list(request.form.keys()),file=sys.stderr)

    # load and prepare the photograph

    image_string = request.form['image']

    image = Image.open(BytesIO(base64.b64decode(image_string)))

    image.save('new_input.jpg',format='JPEG')

    #image = image.resize((224,224),Image.BILINEAR)

    #image.save('input_scaled.jpg',format='JPEG')

    #input_array = np.array(image)
    #cv2.imwrite('array.jpg',input_array)
    
    #input_array = imresize(input_array,(224,224),'nearest')
    #input_array = cv2.resize(input_array,(224,224))
    
    #cv2.imwrite('opencv_resize_array.jpg',input_array)

    #input_array = np.expand_dims(input_array,axis=0)

    #print("Array Shape = ",input_array.shape,file=sys.stderr)
    
    photo = extract_features('new_input.jpg')

    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    
    K.clear_session();

    print(description,file=sys.stderr)
    
    return description


if(__name__ == "__main__"):
    app.run(host = '0.0.0.0',port = 5001)

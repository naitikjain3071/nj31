import os
import flask
import io
from skimage.transform import resize

from PIL import Image 
import base64
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import tensorflow as tf
#import tensorflow.compat.v1.keras.backend as tb
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import numpy as np
import pickle
import cv2
from flask import Flask
import os
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify

app = Flask(__name__)


THIS_FOLDER = os.path.abspath(os.path.dirname(__file__))
my_file = os.path.join(THIS_FOLDER, 'final97.h5')
print(my_file)

global sess
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
global model
model = load_model(my_file)
global graph
graph = tf.compat.v1.get_default_graph()
THIS_FOLDER = os.path.abspath(os.path.dirname(__file__))
my_file = os.path.join(THIS_FOLDER, 'plant_disease_label_transform (2).pkl')
image_labels = pickle.load(open(my_file, 'rb'))
print("Hey")
EPOCHS = 25
STEPS = 100
LR = 1e-3
BATCH_SIZE = 32
WIDTH = 128
HEIGHT = 128
DEPTH = 3
DEFAULT_IMAGE_SIZE = tuple((128, 128))
print(" * Model loaded!")

@app.route("/predict", methods=["POST"])
def prediction():
    
    file = request.files['file']
    print(file)
    file.save(r'test.jpg')
    # Step 1
    # try:
    #     #image = cv2.imread(image_dir)
    #     if image is not None:
    #         image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
    #         return img_to_array(image)
    #     else:
    #         return np.array([])
    # except Exception as e:
    #     print(f"Error : {e}")
    #     return None
    # my_image = plt.imread(os.path.join('/content/drive/My Drive/flask/uploads', filename))

    # Step 2
    my_image = plt.imread(r'test.jpg')
    if my_image is not None:
            image_array = cv2.resize(my_image, DEFAULT_IMAGE_SIZE)
            image_array = img_to_array(image_array)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    #np_image=np_image[::-1][::1][::1][::1]
    np_image = np_image[:,:,:,::-1]
    #plt.imshow(plt.imread(image_path))
    # for i in range(0,len(np_image)):
    #     for j in range(0,len(np_image[i])): 
    #         for k in range(0,len(np_image[j])):
    #             for l in range(0,len(np_image[k])):
    #                 print(np_image[i][j][k][l])
    result = model.predict_classes(np_image)
    #print(np_image)
    #print((image_labels.classes_[result][0]))
    z=image_labels.classes_[result][0]
    msg={'class':z}
    print(msg)
    os.remove(r'test.jpg')
    print(np_image)
    print(np_image.shape)
    return jsonify(msg)
app.run()
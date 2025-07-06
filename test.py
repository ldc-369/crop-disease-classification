import os
import numpy as np
from werkzeug.utils import secure_filename

from tensorflow import convert_to_tensor
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNEL = 3

classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# model = keras.models.load_model('./model/model.keras')

# def validate_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def prepare_input(file_paths):
    X_predict = np.empty((len(file_paths), IMAGE_SIZE, IMAGE_SIZE, CHANNEL), dtype=np.uint8) # uint8: 0-255

    for file_path in file_paths:
        PIL_img = load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)) # load and resize
        img_arr = img_to_array(PIL_img) / 255.0  # scale pixel về [0, 1]
        img_arr = img_arr.astype("float32")
        
        X_predict = np.concatenate((X_predict, np.expand_dims(img_arr, axis=0)), axis=0) # 4D (batch_size, H, W, C)
    
    X_predict = X_predict[1:]
    return convert_to_tensor(X_predict)
        
file_names = ["Potato_healthy-26-_0_8117.jpg"]
file_paths = [os.path.join('dataset\\Potato\\Potato___healthy', file_name) for file_name in file_names]

X_predict = prepare_input(file_paths)
print(X_predict)
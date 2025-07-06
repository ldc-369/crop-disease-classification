import os
import numpy as np
from werkzeug.utils import secure_filename

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNEL = 3

model = load_model('./model/model.keras')
classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def validate_file(file_name):
    if '.' in file_name and file_name.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}:
        return file_name

file_names = ["f528e881-b237-4797-ab54-0152eb901f53___RS_LB 3854.JPG", "f2676f08-798e-4291-8aa2-0655ca54cb07___RS_LB 4292.JPG", "e26ad557-11c8-44fd-aad1-dea51c613742___RS_LB 5115.JPG"]
file_paths = [os.path.join('dataset\\test', validate_file(file_name)) for file_name in file_names]

def prepare_input(file_paths):
    X_predict = np.empty((len(file_paths), IMAGE_SIZE, IMAGE_SIZE, CHANNEL), dtype=np.uint8) # uint8: 0-255

    for file_path in file_paths:
        PIL_img = load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)) # load and resize
        
        img_arr = img_to_array(PIL_img) / 255.0  # scale pixel về [0, 1]
        img_arr = img_arr.astype("float32")
        
        X_predict = np.concatenate((X_predict, np.expand_dims(img_arr, axis=0)), axis=0)[1:] # 4D (batch_size, H, W, C)
    
    return convert_to_tensor(X_predict)
    
def predict(X_predict):
    Y_predict = model.predict(X_predict) # 2D
    
    predicted_label = [classes[np.argmax(item)] for item in Y_predict] # lấy chỉ mục của giá trị lớn nhất

    predicted_probability = [round(np.max(item) * 100, 2) for item in Y_predict] # lấy giá trị lớn nhất

    return predicted_label, predicted_probability

X_predict = prepare_input(file_paths)
predicted_label, predicted_probability = predict(X_predict)

print(predicted_label)
print(predicted_probability)
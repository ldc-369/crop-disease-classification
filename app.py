from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np



app = Flask(__name__)

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNEL = 3

labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

model = ks.models.load_model('./crop-disease-classification/model/model.h5')

def validate_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def prepare_input(filepath):
    img_PIL = ks.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # load and resize
    
    # img_tf = tf.image.convert_image_dtype(img_tf, tf.float32)   # convert_image_dtype() tư động scale
    img_arr = ks.preprocessing.image.img_to_array(img_PIL)   # 3D (h, w, rgb)
    
    img_arr = tf.expand_dims(img_arr, axis=0)  # 4D (batch_size, h, w, rgb)
    F_predict = tf.convert_to_tensor(img_arr)
    
    return F_predict

def predict(F_predict):
    y_predict = model.predict(F_predict) # return 2D
    
    predicted_label = labels[np.argmax(y_predict[0])]  # lấy chỉ mục của giá trị lớn nhất

    predicted_score = round(np.max(y_predict[0]) * 100, 2)

    return [predicted_label, predicted_score]

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file_field' not in request.files:
            return render_template('index.html', message='No file found')  # render trang index.html với biến message
        
        file = request.files['file_field'] 

        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file and validate_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('crop-disease-classification\\static', filename)
            
            file.save(filepath)

            F_predict = prepare_input(filepath)
            predicted_label, predicted_score = predict(F_predict)
            
            return render_template('index.html', image_path=f"static/{filename}", actual_label=predicted_label, predicted_label=predicted_label, predicted_score=predicted_score)
    
    return render_template("index.html")  # render index.html

if __name__ == '__main__':
    app.run(debug=True)

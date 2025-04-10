from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np



app = Flask(__name__)

BATCH_SIZE = 64
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 20

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

model = tf.keras.models.load_model('./model/model.h5')


def validate_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict(img):
    # chuyển ảnh đơn lẽ (h, w, rgb) sang (batch_size, h, w, rgb) (tính theo batch_size (nhiều ảnh))
    array_input = tf.keras.preprocessing.image.img_to_array(img) # 1D
    array_input = tf.expand_dims(array_input, 0)   # thêm chiều

    y_predicted = model.predict(array_input) # return 2D

    predicted_label = class_names[np.argmax(y_predicted[0])]  # lấy chỉ mục của giá trị lớn nhất

    predicted_score = round(100 * (np.max(y_predicted[0])), 2) # chuyển về %

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
            filepath = os.path.join('static', filename)
            
            file.save(filepath)  # lưu file

            # Read the image
            img_tf = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            predicted_label, predicted_score = predict(img_tf)

            return render_template('index.html', image_path=filepath, actual_label=predicted_label, predicted_label=predicted_label, predicted_score=predicted_score)
    
    return render_template("index.html")  # render trang index.html

if __name__ == '__main__':
    app.run(debug=True)
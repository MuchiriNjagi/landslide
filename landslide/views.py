from flask import Blueprint, render_template
from flask_login import login_required, current_user
from flask import Flask, request, render_template, jsonify
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

views = Blueprint('views', __name__)

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

model = load_model(r"C:\Users\Administrator\Desktop\landslide detection system\landslide\model1_save.h5", custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

def preprocess_hdf5_image(image_path):
    f_data = np.zeros((1, 128, 128, 6))

    with h5py.File(image_path, 'r') as hdf:
        data = np.array(hdf.get('img'))

        data[np.isnan(data)] = 0.000001
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))

        f_data[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb
        f_data[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb
        f_data[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb
        f_data[0, :, :, 3] = data_ndvi
        f_data[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope
        f_data[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation

    uploaded_image = data[:, :, 1:4]
    uploaded_image = (uploaded_image - uploaded_image.min()) / (uploaded_image.max() - uploaded_image.min()) * 255
    uploaded_image = uploaded_image.astype(np.uint8)

    return f_data, uploaded_image

def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image)

    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    f_data = np.zeros((1, 128, 128, 6))
    mid_rgb = image_array.max() / 2.0

    f_data[0, :, :, 0] = 1 - image_array[:, :, 0] / mid_rgb
    f_data[0, :, :, 1] = 1 - image_array[:, :, 1] / mid_rgb
    f_data[0, :, :, 2] = 1 - image_array[:, :, 2] / mid_rgb
    f_data[0, :, :, 3] = 0
    f_data[0, :, :, 4] = 0
    f_data[0, :, :, 5] = 0
    
    print(f"Image shape: {image_array.shape}")
    print(f"Preprocessed data shape: {f_data.shape}")
    print(f"Preprocessed data: {f_data[0, :, :, :3].mean(axis=(0, 1))}")

    return f_data, image_array

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename.endswith('.h5'):
        processed_data, uploaded_image = preprocess_hdf5_image(file)
    else:
        image = Image.open(file.stream)
        processed_data, uploaded_image = preprocess_image(image)

    prediction = model.predict(processed_data)
    print("Raw Prediction:", prediction)
    
    color_coded_prediction = np.zeros((prediction.shape[1], prediction.shape[2], 3), dtype=np.uint8)
    
    # Black for values below 0.1
    black_mask = prediction[0, :, :, 0] < 0.1
    color_coded_prediction[black_mask] = [0, 0, 0]
    
    # Green for values between 0.1 and 0.3
    green_mask = (prediction[0, :, :, 0] >= 0.1) & (prediction[0, :, :, 0] < 0.3)
    color_coded_prediction[green_mask] = [0, 255, 0]
    
    # Red for values greater than or equal to 0.3
    red_mask = prediction[0, :, :, 0] >= 0.3
    color_coded_prediction[red_mask] = [255, 0, 0]
    
    print("Color-Coded Prediction:", color_coded_prediction)

    # Convert to byte stream to send as response
    uploaded_image_io = io.BytesIO()
    prediction_image_io = io.BytesIO()
    Image.fromarray(uploaded_image).save(uploaded_image_io, format='PNG')
    Image.fromarray(color_coded_prediction).save(prediction_image_io, format='PNG')
    uploaded_image_io.seek(0)
    prediction_image_io.seek(0)

    uploaded_image_base64 = base64.b64encode(uploaded_image_io.getvalue()).decode('utf-8')
    prediction_image_base64 = base64.b64encode(prediction_image_io.getvalue()).decode('utf-8')

    return jsonify({
        'uploaded_image': f'data:image/png;base64,{uploaded_image_base64}',
        'prediction_image': f'data:image/png;base64,{prediction_image_base64}'
    })

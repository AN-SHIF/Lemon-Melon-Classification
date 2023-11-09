# app.py

import os
from flask import Flask, render_template, request, jsonify
import joblib
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained SVM classifier
model = joblib.load('model.pkl')  # Replace with the path to your trained model

# Define an upload folder for images
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((120, 120))  # Resize to match your training data size
    img = np.array(img)
    img = img.reshape(1, -1)
    img = img / 255.0  # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)

        # Define the class labels
        class_labels = {0: 'Lemon', 1: 'Melon'}

        result = class_labels[prediction[0]]

        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

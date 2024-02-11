from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS

model_path = 'best_model_improved'
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Convert the FileStorage object to a BytesIO object
    img_bytes = BytesIO(file.read())

    # Process the file
    img = image.load_img(img_bytes, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    prediction = model.predict(img_array)
    threshold = 0.5
    result = "PNEUMONIA" if prediction[0] >= threshold else "NORMAL"
    
    return jsonify({'classification': result})

if __name__ == '__main__':
    app.run(debug=True)

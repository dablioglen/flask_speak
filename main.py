import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

inv_label_map = {str(v): k for k, v in label_map.items()}

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='sign_language_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_landmarks(landmarks):
    try:
        landmarks = np.array(landmarks, dtype=np.float32).reshape(1, 21, 3)
        interpreter.set_tensor(input_details[0]['index'], landmarks)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        
        print(f"Predicted class: {predicted_class}")  # Debug print
        print(f"Output data: {output_data}")  # Debug print
        
        if str(predicted_class) in inv_label_map:
            predicted_label = inv_label_map[str(predicted_class)]
            confidence = float(output_data[0][predicted_class])
            print(f"Predicted label: {predicted_label}, Confidence: {confidence}")  # Debug print
            return {'label': predicted_label, 'confidence': confidence}
        else:
            print(f"Unknown class: {predicted_class}")  # Debug print
            return {'label': 'Unknown', 'confidence': 0.0}
    except Exception as e:
        print(f"Error in predict_landmarks: {str(e)}")  # Debug print
        return {'label': 'Error', 'confidence': 0.0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = data.get('landmarks')
    if not landmarks:
        return jsonify({'error': 'No landmarks provided'}), 400
    prediction = predict_landmarks(landmarks)
    print(f"Returning prediction: {prediction}")  # Debug print
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)

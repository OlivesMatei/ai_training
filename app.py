from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model (assume model is saved as 'disease_model.h5')
model = tf.keras.models.load_model('disease_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = request.form.to_dict()
    features = [float(data[feature]) for feature in data]
    
    # Scale features (use the same scaler used during training)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])
    
    # Make predictions
    predictions = model.predict(features_scaled)
    predictions = [int(pred[0] > 0.5) for pred in predictions]  # Convert probabilities to binary outcome
    
    return jsonify({
        'diabetes': predictions[0],
        'heart_disease': predictions[1],
        'alzheimer': predictions[2]
    })

if __name__ == '__main__':
    app.run(debug=True)

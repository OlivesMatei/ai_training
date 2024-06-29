from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import data_preprocessing as dp

app = Flask(__name__)

model = tf.keras.models.load_model('disease_model.h5')
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = np.load('scaler_params.npy')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    result = (prediction > 0.5).astype(int).tolist()[0]
    return jsonify({
        'diabetes': result[0],
        'heart_disease': result[1],
        'alzheimer': result[2]
    })

if __name__ == '__main__':
    app.run(debug=True)

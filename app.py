from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')  # Path to the saved model

# Load the scaler and label encoders
scaler = StandardScaler()
scaler.scale_ = np.load('scaler.npy')  # Path to the saved scaler

label_encoder_city = LabelEncoder()
label_encoder_city.classes_ = np.load('city_classes.npy', allow_pickle=True)  # Path to the saved city classes

label_encoder_country = LabelEncoder()
label_encoder_country.classes_ = np.load('country_classes.npy', allow_pickle=True)  # Path to the saved country classes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    city = data['city']
    country = data['country']
    latitude = data['latitude']
    longitude = data['longitude']
    year = data['year']
    month = data['month']
    day = data['day']
    
    city_encoded = label_encoder_city.transform([city])[0]
    country_encoded = label_encoder_country.transform([country])[0]
    features = np.array([[city_encoded, country_encoded, latitude, longitude, year, month, day]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    predicted_temp = prediction[0][0]
    
    return jsonify({'predicted_temperature': predicted_temp})

if __name__ == '__main__':
    app.run(debug=True)


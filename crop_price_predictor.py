from flask import Flask, request, jsonify
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)

# Placeholder function to fetch crop price data
def fetch_crop_price_data():
    data = {
        'crop_name': ['wheat', 'rice', 'corn'],
        'price': [20, 30, 25]
    }
    return pd.DataFrame(data)

# Function to train and save the model
def train_model():
    data = fetch_crop_price_data()
    X = pd.get_dummies(data[['crop_name']], drop_first=True)
    y = data['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    joblib.dump(model, 'crop_price_model.pkl')
    joblib.dump(list(X.columns), 'model_features.pkl')  # Save feature names

# Train model only if not already trained
if not os.path.exists('crop_price_model.pkl'):
    train_model()

# Endpoint to predict crop price
@app.route('/predict_price', methods=['GET'])
def predict_price():
    crop_name = request.args.get('crop_name')
    
    if not crop_name:
        return jsonify({"error": "Please provide a crop name"}), 400

    model = joblib.load('crop_price_model.pkl')
    feature_names = joblib.load('model_features.pkl')

    # Prepare input data with consistent columns
    input_data = pd.DataFrame({'crop_name': [crop_name]})
    input_data = pd.get_dummies(input_data)

    # Align input features with model features
    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0  # Add missing columns

    input_data = input_data[feature_names]  # Ensure column order

    predicted_price = model.predict(input_data)

    return jsonify({"crop_name": crop_name, "predicted_price": predicted_price[0]})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)

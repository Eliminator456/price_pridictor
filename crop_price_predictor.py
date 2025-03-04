from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
import re
from serpapi import GoogleSearch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Load API key from Render environment

# Function to fetch live crop price from SerpAPI
def fetch_crop_price(crop_name):
    params = {
        "engine": "google",
        "q": f"{crop_name} price per kg in India",
        "api_key": SERPAPI_KEY
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    for result in results.get("organic_results", []):
        snippet = result.get("snippet", "")

        # Extract ₹ price correctly using regex
        match = re.search(r"₹\s?(\d+)", snippet)
        if match:
            return float(match.group(1))  # Convert extracted price to float

    return None  # No price found

# Function to fetch training data (Placeholder dataset)
def fetch_crop_price_data():
    data = {
        'crop_name': ['wheat', 'rice', 'corn', 'barley'],
        'price': [20, 30, 25, 28]  # Prices in ₹ per kg (sample data)
    }
    return pd.DataFrame(data)

# Train and save the model
def train_model():
    data = fetch_crop_price_data()
    X = pd.get_dummies(data[['crop_name']], drop_first=True)
    y = data['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    joblib.dump(model, 'crop_price_model.pkl')
    joblib.dump(list(X.columns), 'model_features.pkl')  # Save feature names

# Ensure model is trained before starting
if not os.path.exists('crop_price_model.pkl'):
    train_model()

# Prediction API
@app.route('/predict_price', methods=['GET'])
def predict_price():
    crop_name = request.args.get('crop_name')

    if not crop_name:
        return jsonify({"error": "Please provide a crop name"}), 400

    # Load model & feature names
    model = joblib.load('crop_price_model.pkl')
    feature_names = joblib.load('model_features.pkl')

    # Prepare input data
    input_data = pd.DataFrame({'crop_name': [crop_name]})
    input_data = pd.get_dummies(input_data)

    # Ensure all required columns are present
    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0  # Add missing columns

    input_data = input_data[feature_names]  # Ensure correct order

    # If all values are 0 (unknown crop), return a default response
    if input_data.sum().sum() == 0:
        return jsonify({"crop_name": crop_name, "predicted_price": "Unknown crop"}), 400

    # Predict price
    predicted_price = model.predict(input_data)[0]

    # Fetch live price from SerpAPI
    live_price = fetch_crop_price(crop_name)

    return jsonify({
        "crop_name": crop_name,
        "predicted_price": round(predicted_price, 2),
        "live_price": live_price if live_price else "Not Available"
    })

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)

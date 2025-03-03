from flask import Flask, request, jsonify
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Function to fetch crop price data (placeholder for actual implementation)
def fetch_crop_price_data():
    # This function should implement data collection logic
    # For now, we will return a placeholder DataFrame
    data = {
        'crop_name': ['wheat', 'rice', 'corn'],
        'price': [20, 30, 25]  # Placeholder prices
    }
    return pd.DataFrame(data)

# Function to train the model (placeholder for actual implementation)
def train_model():
    data = fetch_crop_price_data()
    X = data[['crop_name']]  # Features
    y = data['price']  # Target variable
    
    # Encode categorical data (placeholder)
    X = pd.get_dummies(X, drop_first=True)
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'crop_price_model.pkl')

# Endpoint to predict crop price
@app.route('/predict_price', methods=['GET'])
def predict_price():
    crop_name = request.args.get('crop_name')
    
    if not crop_name:
        return jsonify({"error": "Please provide a crop name"}), 400
    
    # Load the model
    model = joblib.load('crop_price_model.pkl')
    
    # Prepare input for prediction (placeholder)
    input_data = pd.DataFrame({'crop_name': [crop_name]})
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Make prediction (placeholder)
    predicted_price = model.predict(input_data)
    
    return jsonify({"crop_name": crop_name, "predicted_price": predicted_price[0]})

if __name__ == '__main__':
    train_model()  # Train the model when the app starts
    app.run(debug=True)

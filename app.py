"""
Flask Web Application for House Price Prediction
=================================================
Interactive web interface for predicting house prices using the linear regression model.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store the trained model and model info
model = None
model_info = {}


def load_and_train_model():
    """Load data and train the model."""
    global model, model_info
    
    # Load data
    df = pd.read_csv('house_prices.csv')
    df = df.dropna()
    
    # Prepare data
    X = df[['House Size (sq ft)']].values
    y = df['Price (in $1000)'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate model metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Store model information
    model_info = {
        'intercept': float(model.intercept_),
        'slope': float(model.coef_[0]),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'equation': f'Price = {model.intercept_:.2f} + {model.coef_[0]:.4f} × House Size',
        'data_points': len(df)
    }
    
    print("Model loaded and trained successfully!")


@app.route('/')
def index():
    """Serve the main page (works for both Flask and static hosting)."""
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction."""
    try:
        data = request.get_json()
        house_size = float(data.get('house_size', 0))
        
        if house_size <= 0:
            return jsonify({'error': 'House size must be greater than 0'}), 400
        
        # Make prediction
        predicted_price_k = model.predict([[house_size]])[0]
        predicted_price = predicted_price_k * 1000
        
        return jsonify({
            'success': True,
            'house_size': house_size,
            'predicted_price_k': round(predicted_price_k, 2),
            'predicted_price': round(predicted_price, 2),
            'formatted_price_k': f'${predicted_price_k:,.2f}K',
            'formatted_price': f'${predicted_price:,.2f}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """API endpoint to get model information."""
    return jsonify(model_info)


if __name__ == '__main__':
    # Load and train model when app starts
    load_and_train_model()
    print("\n" + "="*60)
    print("HOUSE PRICE PREDICTION WEB APP")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)


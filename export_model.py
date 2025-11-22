"""
Export Model Parameters for Static HTML Version
================================================
This script trains the model and exports its parameters to a JSON file
that can be used in a static HTML/JavaScript version.
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from linear_regression_model import load_data, prepare_data, train_model

def export_model_parameters():
    """Train model and export parameters to JSON."""
    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Prepare model data
    model_data = {
        'intercept': float(model.intercept_),
        'slope': float(model.coef_[0]),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'equation': f'Price = {model.intercept_:.2f} + {model.coef_[0]:.4f} × House Size',
        'data_points': len(df),
        'min_size': float(X.min()),
        'max_size': float(X.max())
    }
    
    # Save to JSON file
    with open('model_params.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print("="*60)
    print("MODEL PARAMETERS EXPORTED")
    print("="*60)
    print(f"✓ Model parameters saved to 'model_params.json'")
    print(f"\nModel Equation: {model_data['equation']}")
    print(f"R² Score: {model_data['r2']:.4f} ({model_data['r2']*100:.2f}%)")
    print(f"RMSE: ${model_data['rmse']:.2f}K")
    print(f"Data Points: {model_data['data_points']}")
    print("="*60)
    
    return model_data

if __name__ == "__main__":
    export_model_parameters()


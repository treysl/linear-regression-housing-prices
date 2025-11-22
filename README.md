# House Price Prediction - Linear Regression Model

This project implements a simple linear regression model to predict house prices based on house size using historical real estate data.

## Overview

The model analyzes the relationship between house size (in square feet) and price (in thousands of dollars) to help a real estate agency:
- Set competitive property prices
- Assist clients in making informed purchasing decisions
- Predict prices for new listings based on house size

## Dataset

The dataset (`house_prices.csv`) contains:
- **House Size (sq ft)**: The size of the house in square feet
- **Price (in $1000)**: The price of the house in thousands of dollars

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train the model and generate predictions:

```bash
python linear_regression_model.py
```

The script will:
1. Load and explore the housing data
2. Train a linear regression model
3. Evaluate model performance (R², RMSE, MAE)
4. Generate a visualization showing the regression line
5. Provide example predictions for different house sizes

## Model Output

The script generates:
- **Console output**: Model metrics, equation, and example predictions
- **Visualization**: A scatter plot with the regression line saved as `regression_plot.png`

## Making Predictions

After running the script, you can use the model to predict prices:

```python
from linear_regression_model import predict_price, load_data, prepare_data, train_model

# Load and prepare data
df = load_data()
X, y = prepare_data(df)
model = train_model(X, y)

# Predict price for a 2000 sq ft house
predicted_price = predict_price(model, 2000)
print(f"Predicted price: ${predicted_price:.2f}K (${predicted_price*1000:,.2f})")
```

## Model Interpretation

The linear regression model provides:
- **Intercept**: Base price when house size is 0 (theoretical)
- **Slope**: Price increase per square foot
- **R² Score**: Percentage of price variance explained by house size
- **RMSE**: Average prediction error in thousands of dollars

## Files

- `linear_regression_model.py`: Main script with model implementation
- `house_prices.csv`: Dataset with house sizes and prices
- `requirements.txt`: Python dependencies
- `regression_plot.png`: Generated visualization (created after running the script)



# filepath: /c:/Users/darin/Code/California Housing ML/app.py
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, request, jsonify

# Loads dataset from sklearn
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Preprocess the data
X = df.drop(columns=['PRICE'])  # Features
y = df['PRICE']  # Target variable

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Trains the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Evaluates the model on MAE, MSE, and R²
y_pred = model.predict(X_test)

print("Model Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

# Deploys the model as a Flask API
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Use a POST request to send data to /predict."
        
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    prediction = max(prediction[0], 0)  # Makes sure the price is never negative
    return jsonify({"predicted_price": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
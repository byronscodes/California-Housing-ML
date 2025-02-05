# California Housing Machine Learning Model

This project uses a **Dockerfile** to build a virtual environment that installs the necessary dependencies, including **NumPy, Pandas, Pickle, Scikit-Learn, and Flask**.

## Overview
The Python script fetches a built-in **California housing dataset** from `sklearn.datasets`, which includes various house-related features. It then trains a **Linear Regression model** to predict house prices (in hundreds of thousands of dollars). The model's accuracy is evaluated using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score**.

The trained model is saved using **Pickle**, so it does not need to be retrained for every API request. **Flask** creates a REST API that handles HTTP requests, making it easy to get predictions via a simple POST request.

## Features Used for Prediction
The model takes in **eight numerical features**:
1. **Median income** *(in tens of thousands of dollars)*
2. **House age** *(years)*
3. **Average number of rooms per house**
4. **Average number of bedrooms per house**
5. **Block population**
6. **Block latitude** *(in degrees)*
7. **Block longitude** *(in degrees)*

## API Usage
After running the Flask app, you can send a **POST request** with housing data to get a price prediction.

### **Example Request (cURL)**
```sh
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
         }'
```

## **Running the App**

### **Starting Flask API Locally**
```sh
python app.py
```

### **Starting Through Dockerfile**
```sh
docker build -t housing-api .
docker run -p 5000:5000 housing-api
```

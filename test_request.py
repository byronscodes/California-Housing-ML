import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [8.0, 41.0, 4.0, 1.0, 0.6, 5.5, 2.5, -118.0]}

response = requests.post(url, json=data)
print(response.json())  # Expected output: {'predicted_price': some_value}
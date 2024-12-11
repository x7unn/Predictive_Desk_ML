import requests
import json

# Define the API endpoint URL
url = "http://127.0.0.1:5000/predict_ticket_volume"

# Define the input data
data = {
    "ticket_data": {
        "2024-10-01": 50,
        "2024-10-02": 60,
        "2024-10-03": 55,
        "2024-10-04": 45,
        "2024-10-05": 70,
        "2024-10-06": 80,
        "2024-10-07": 65,
        "2024-10-08": 60,
        "2024-10-09": 75,
        "2024-10-10": 85,
        "2024-10-11": 50,
        "2024-10-12": 60,
        "2024-10-13": 55,
        "2024-10-14": 45,
        "2024-10-15": 70,
        "2024-10-16": 80,
        "2024-10-17": 65,
        "2024-10-18": 60,
        "2024-10-19": 75,
        "2024-10-20": 85,
        "2024-10-21": 50,
        "2024-10-22": 60,
        "2024-10-23": 55,
        "2024-10-24": 45,
        "2024-10-25": 70,
        "2024-10-26": 80,
        "2024-10-27": 65,
        "2024-10-28": 60,
        "2024-10-29": 75,
        "2024-10-30": 85
    },
    "holidays": {
        "2024-10-05": "National Holiday",
        "2024-10-12": "Regional Holiday"
    },
    "product_launches": {
        "2024-10-08": "Minor Product Launch",
        "2024-10-15": "Major Product Launch"
    },
    "range":"monthly"
}

# Make the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

# Check if the request was successful
if response.status_code == 200:
    print("Prediction results:", response.json())
else:
    print(f"Error: {response.__dict__}")


# URL of your API endpoint (replace with the actual endpoint URL)
api_url = "http://127.0.0.1:5000/predict_ticket_volume_using_external"

def test_api_predictions(current_date, range_type):
    # Prepare the payload for the API request
    payload = {
        "current_date": current_date,
        "date_range": range_type,  # 'weekly' or 'monthly'
    }

    # Set headers (optional, if needed by your API, e.g., for JSON)
    headers = {'Content-Type': 'application/json'}

    # Make the POST request to the API
    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        # Parse the response JSON
        predictions = response.json()
        
        # Print out the predictions
        print(f"Predictions for {range_type} range starting from {current_date}:")
        for prediction in predictions:
            print(f"Date: {prediction['date']}, Predicted Ticket Volume: {prediction['forecasted_value']}")
    else:
        print(f"Failed to get predictions. Status Code: {response.status_code}")

# Set the current date for testing (replace with the actual current date or a fixed date)
current_date = datetime(2024, 11, 27).strftime('%Y-%m-%d')  # Example: "2024-11-27"

# Test the API for weekly predictions
test_api_predictions(current_date, "weekly")

# Test the API for monthly predictions
test_api_predictions(current_date, "monthly")

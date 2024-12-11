from flask import Flask, request, jsonify
from util import predict_resolution_time, make_volume_predictions, make_volume_predictions_using_external_data
from datetime import datetime

app = Flask(__name__)

@app.route('/predict', methods=['GET'])  # Change this to allow GET requests
def predict():
    try:
    # Extract query parameters from the URL

        print(request.args.__dict__)

        issue_type = request.args.get('Issue Type')
        urgency = request.args.get('Urgency')
        priority = request.args.get('Priority')
        current_ticket_volume = int(request.args.get('Current Ticket Volume'))
        holiday_season = int(request.args.get('Holiday Season'))
        product_launch_near = int(request.args.get('Product Launch Near'))

        # Prepare input data as a dictionary
        input_data = {
            'Issue Type': issue_type,
            'Urgency': urgency,
            'Priority': priority,
            'Current Ticket Volume': current_ticket_volume,
            'Holiday Season': holiday_season,
            'Product Launch Near': product_launch_near
        }

        print(input_data)
        # Make prediction
        predictions = predict_resolution_time(input_data)
        return jsonify({'resolution_time': predictions.tolist()[0]})  # Return predictions as JSON
    except Exception as e:
        print(e)
        return jsonify({'resolution_time': -1})  # Return predictions as JSON

# API endpoint to make ticket volume predictions
@app.route('/predict_ticket_volume', methods=['POST'])
def predict_ticket_volume():
    try:
        # Parse input JSON data
        data = request.get_json()

        ticket_data = data['ticket_data']  # Dictionary with dates and ticket counts
        holidays = data['holidays']  # Dictionary with holiday dates and names
        product_launches = data['product_launches']  # Dictionary with product launch dates and names
        range = data['range']  # Dictionary with product launch dates and names

        # Ensure past 30 days ticket data is provided
        if len(ticket_data) != 30:
            return jsonify({'error': 'You must provide ticket data for the past 30 days.'}), 400

        # Call the prediction function
        predictions = make_volume_predictions(ticket_data, holidays, product_launches, range)

        # Return the prediction results as JSON
        return jsonify(predictions.to_dict(orient='records')), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_ticket_volume_using_external', methods=['POST'])
def predict_ticket_volume_external():
    try:
        # Get the current date and range from the request
        current_date = request.json.get('current_date', str(datetime.now().date()))
        date_range = request.json.get('date_range')  # 'weekly' or 'monthly'

        predictions = make_volume_predictions_using_external_data(current_date, date_range)

        # Return the prediction results as JSON
        return jsonify(predictions.to_dict(orient='records')), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000)


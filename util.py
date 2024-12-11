from prophet import Prophet
from datetime import datetime, timedelta
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved pipeline
model = joblib.load('./model/ticket_resolution_model.pkl')
external_data = pd.read_csv('./training/data/external_customer_support_tickets.csv')

def predict_resolution_time(input_data):
    input_df = pd.DataFrame([input_data])
    predictions = model.predict(input_df)
    return predictions

# Function to calculate proximity to nearest holiday or product launch
def get_proximity_to_event(date, event_dates):
    before = event_dates[event_dates <= date]
    after = event_dates[event_dates > date]
    
    closest_before = before.max() if not before.empty else pd.Timestamp.min
    closest_after = after.min() if not after.empty else pd.Timestamp.max

    days_before = (date - closest_before).days if closest_before != pd.Timestamp.min else float('inf')
    days_after = (closest_after - date).days if closest_after != pd.Timestamp.max else float('inf')
    
    proximity = min(days_before, days_after)
    return proximity if proximity <= 14 else 0

# Function to create the model and make predictions
def make_volume_predictions(ticket_data, holidays, product_launches, range='weekly'):
    # Determine parameters based on range
    if range == 'weekly':
        past_days = 30
        future_days = 7
        rolling_window = 7
    elif range == 'monthly':
        past_days = 90
        future_days = 30
        rolling_window = 30
    else:
        raise ValueError("Invalid range specified. Use 'weekly' or 'monthly'.")

    # Filter ticket data for the past period based on range
    ticket_df = pd.DataFrame(list(ticket_data.items()), columns=["ds", "y"])
    ticket_df["ds"] = pd.to_datetime(ticket_df["ds"])
    cutoff_date = ticket_df["ds"].max() - timedelta(days=past_days)
    ticket_df = ticket_df[ticket_df["ds"] > cutoff_date]

    # Add external factors (holidays and product launches)
    ticket_df["is_holiday"] = ticket_df["ds"].astype(str).map(holidays).notnull().astype(int)
    ticket_df["is_product_launch"] = ticket_df["ds"].astype(str).map(product_launches).notnull().astype(int)

    # Add time-based features
    ticket_df["day_of_week"] = ticket_df["ds"].dt.dayofweek
    ticket_df["is_weekend"] = ticket_df["day_of_week"].isin([5, 6]).astype(int)
    ticket_df["month"] = ticket_df["ds"].dt.month
    ticket_df["quarter"] = ticket_df["ds"].dt.quarter

    # Add rolling features
    ticket_df[f"rolling_mean_{rolling_window}"] = ticket_df["y"].rolling(window=rolling_window).mean().fillna(method='bfill')

    # Add proximity features
    ticket_df["holiday_proximity"] = ticket_df["ds"].apply(lambda x: get_proximity_to_event(x, pd.to_datetime(list(holidays.keys()))))
    ticket_df["product_launch_proximity"] = ticket_df["ds"].apply(lambda x: get_proximity_to_event(x, pd.to_datetime(list(product_launches.keys()))))

    # Create Prophet model with external regressors
    prophet = Prophet()
    prophet.add_regressor("is_holiday")
    prophet.add_regressor("is_product_launch")
    prophet.add_regressor("is_weekend")
    prophet.add_regressor("holiday_proximity")
    prophet.add_regressor("product_launch_proximity")
    prophet.add_regressor(f"rolling_mean_{rolling_window}")
    prophet.fit(ticket_df)

    # Prepare future data for prediction
    future_dates = pd.date_range(start=ticket_df["ds"].max() + timedelta(days=1), periods=future_days)
    future_data = pd.DataFrame({"ds": future_dates})

    # Add external factors for future data
    future_data["is_holiday"] = future_data["ds"].astype(str).map(holidays).notnull().astype(int)
    future_data["is_product_launch"] = future_data["ds"].astype(str).map(product_launches).notnull().astype(int)
    future_data["day_of_week"] = future_data["ds"].dt.dayofweek
    future_data["is_weekend"] = future_data["day_of_week"].isin([5, 6]).astype(int)
    future_data["month"] = future_data["ds"].dt.month
    future_data["quarter"] = future_data["ds"].dt.quarter

    # Add proximity features for future data
    future_data["holiday_proximity"] = future_data["ds"].apply(lambda x: get_proximity_to_event(x, pd.to_datetime(list(holidays.keys()))))
    future_data["product_launch_proximity"] = future_data["ds"].apply(lambda x: get_proximity_to_event(x, pd.to_datetime(list(product_launches.keys()))))

    # Add rolling mean for future data using the mean of the last rolling_window days of available data
    last_mean = ticket_df["y"].iloc[-rolling_window:].mean()
    future_data[f"rolling_mean_{rolling_window}"] = last_mean

    # Make the prediction
    forecast = prophet.predict(future_data)
    forecast["ds"] = forecast["ds"].dt.strftime("%a, %d %b %Y")

    forecast = forecast.rename(columns={
        "ds": "date", 
        "yhat": "forecasted_value", 
    })

    columns_to_process = ["forecasted_value"]
    forecast[columns_to_process] = forecast[columns_to_process].clip(lower=0)
    forecast['forecasted_value'] = forecast['forecasted_value'].astype(int)

    return forecast[["date", "forecasted_value"]]

def make_volume_predictions_using_external_data(current_date, date_range='weekly', seed_value=42):

    # Set the seed for reproducibility
    np.random.seed(seed_value)
    
    data = external_data
    
    # Ensure 'First Response Time' is in datetime format
    data['First Response Time'] = pd.to_datetime(data['First Response Time'])
    data['Ticket Type'] = data['Ticket Type'].fillna('Unknown')  # Fill missing Ticket Type with 'Unknown'
    data['Ticket Priority'] = data['Ticket Priority'].fillna('Low')  # Fill missing Ticket Priority with 'Low'
    data['Customer Satisfaction Rating'] = data['Customer Satisfaction Rating'].fillna(0)  # Fill missing Ticket Priority with 'Low'
    data['Ticket Channel'] = data['Ticket Channel'].fillna('Unknown')  # Fill missing Ticket Type with 'Unknown'

    # Convert categorical columns to numeric using Label Encoding
    label_encoder = LabelEncoder()
    data['Ticket Type'] = label_encoder.fit_transform(data['Ticket Type'])
    data['Ticket Priority'] = label_encoder.fit_transform(data['Ticket Priority'])
    data['Ticket Channel'] = label_encoder.fit_transform(data['Ticket Channel'])

    data['Customer Age Group'] = pd.cut(data['Customer Age'], bins=[18, 25, 35, 45, 60, 100], labels=['18-25', '26-35', '36-45', '46-60', '60+'])
    data['Customer Age Group'] = label_encoder.fit_transform(data['Customer Age Group'])

    data['day_of_week'] = data['First Response Time'].dt.dayofweek
    data['month'] = data['First Response Time'].dt.month
    data['day_of_year'] = data['First Response Time'].dt.dayofyear

    # Add a random regressor
    data['random_regressor'] = np.random.randint(0, 1, len(data))

    # Select the relevant columns and rename them for Prophet compatibility
    data = data.rename(columns={'First Response Time': 'ds'})

    # Count tickets per day (Ticket Volume)
    data['y'] = data.groupby('ds')['Ticket ID'].transform('count')
    data = data.drop_duplicates(subset=['ds'])

    # Create the Prophet model
    model = Prophet()
    model.add_seasonality(name='weekly', period=7, fourier_order=8)
    model.add_regressor('Ticket Type')
    model.add_regressor('Ticket Priority')
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    # model.add_regressor('day_of_year')
    model.add_regressor('Ticket Channel')
    model.add_regressor('Customer Age Group')
    model.add_regressor('Customer Satisfaction Rating')
    model.add_regressor('random_regressor')  # Adding random regressor

    # Fit the model
    model.fit(data)
    
    # Define future dates for prediction based on the range (weekly or monthly)
    future_dates = []
    current_date = datetime.strptime(current_date, "%Y-%m-%d")

    if date_range == 'weekly':
        # Predict the next 7 days (weekly)
        for i in range(1, 8):  # 7 days ahead
            future_dates.append(current_date + timedelta(days=i))

    elif date_range == 'monthly':
        # Predict the next 30 days (monthly)
        for i in range(1, 31):  # 30 days ahead
            future_dates.append(current_date + timedelta(days=i))
    
    # Create a DataFrame for future predictions
    future = pd.DataFrame({'ds': pd.to_datetime(future_dates)})

    # Generate random values for the future regressors
    future['Ticket Type'] = np.random.choice(data['Ticket Type'].unique(), len(future))
    future['Ticket Priority'] = np.random.choice(data['Ticket Priority'].unique(), len(future))
    future['month'] = future['ds'].dt.month
    future['random_regressor'] = np.random.randint(0, 1, len(future))
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['day_of_year'] = future['ds'].dt.dayofyear

    # Adding placeholders for other regressors
    future['Ticket Channel'] = 0  # Placeholder, adjust as needed
    future['Customer Age Group'] = 0  # Placeholder, adjust as needed
    future['Customer Satisfaction Rating'] = 0  # Placeholder, adjust as needed

    # Predict future values
    forecast = model.predict(future)
    forecast["ds"] = forecast["ds"].dt.strftime("%a, %d %b %Y")

    forecast = forecast.rename(columns={
        "ds": "date", 
        "yhat": "forecasted_value", 
    })

    # Ensure no negative values
    forecast['forecasted_value'] = round(forecast['forecasted_value']*10,0)
    forecast['forecasted_value'] = forecast['forecasted_value'].astype(int)

    return forecast[["date", "forecasted_value"]]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('data/ticket_resolution_dataset.csv')

# Generate mock data for additional external factors
np.random.seed(42)
df['Current Ticket Volume'] = np.random.randint(50, 500, size=len(df))
df['Holiday Season'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])  # 0: No, 1: Yes
df['Product Launch Near'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])  # 0: No, 1: Yes

# Split the data into features and target
X = df[['Issue Type', 'Urgency', 'Priority', 'Current Ticket Volume', 'Holiday Season', 'Product Launch Near']]
y = df['Resolution Time (Hours)']

# Preprocessing for categorical features
categorical_features = ['Issue Type', 'Urgency', 'Priority']
numeric_features = ['Current Ticket Volume', 'Holiday Season', 'Product Launch Near']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Create a pipeline with XGBoost Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to search
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.001, 0.07, 0.06, 0.05],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.8, 1.0]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Save the entire pipeline (including preprocessing steps and model)
joblib.dump(best_model, 'improved_ticket_resolution_pipeline.pkl')

print("Model training completed and saved as 'improved_ticket_resolution_pipeline.pkl'.")

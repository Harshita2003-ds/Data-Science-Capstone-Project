# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from sklearn import config_context
from sklearn.ensemble import GradientBoostingRegressor

# Load your saved Gradient Boosting model with safe config
with config_context(assume_finite=True):
    try:
        model = joblib.load('Gradient_boosting_model.pkl')
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

# Define your app's title and description
st.title("Car Selling Price Prediction App")
st.write(""" 
### Predict the selling price of a car based on its details.
""")

# Create input fields for user to enter data
brand = st.selectbox('Select Car Brand', ['Toyota', 'Honda', 'Ford', 'Other'])  # Add more brands
model_input = st.text_input('Enter Model')
variant = st.text_input('Enter Variant')
name = st.text_input('Enter Name')
year = st.number_input('Enter the Year of Manufacture', min_value=1990, max_value=2024, step=1)
km_driven = st.number_input('Enter Km Driven', min_value=0)  # Ensure Km Driven is positive
fuel = st.selectbox('Select Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric'])  # Add more fuel types
seller_type = st.selectbox('Select Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Select Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Select Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above'])

# Button to trigger prediction
if st.button('Predict Selling Price'):
    # Prepare the input data in DataFrame format
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model_input],
        'Variant': [variant],
        'Name': [name],
        'Year': [year],
        'Km_Driven': [km_driven],
        'Fuel': [fuel],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    # Check for missing or NaN values
    if input_data.isnull().values.any():
        st.error("Inputs contain missing values. Please correct the input and try again.")
    elif input_data.isin(['', None]).any().any():
        st.error("Please fill in all the details.")
    else:
        # One-hot encode categorical variables
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Ensure the model's expected feature names match the input
        input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # Perform the prediction using the loaded model
        try:
            predicted_price = model.predict(input_data_encoded)[0]
            st.success(f"The predicted selling price is: {predicted_price:.2f}")
        except AttributeError as e:
            st.error(f"An attribute error occurred during prediction: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Saving model (optional, only if retraining is required)
# Set up the Gradient Boosting model
final_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the Gradient Boosting model (uncomment and fit when needed)
# final_model.fit(x, y)

# Save the final model using pickle
try:
    with open('Gradient_boosting_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    st.success("Model saved successfully as 'Gradient_Boosting.pkl'")
except Exception as e:
    st.error(f"An error occurred while saving the model: {e}")

# Load the model
try:
    with open('Gradient_boosting_model.pkl', 'rb') as f:
        load_model = pickle.load(f)
    st.success("Model loaded successfully: Gradient Boosting")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

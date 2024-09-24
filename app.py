# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import config_context
import os  # For file handling
import pickle  # For saving/loading model

# Load your saved Random Forest model with safe config
with config_context(assume_finite=True):
    try:
        model = joblib.load('Random_Forest_compressed.pkl')
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
    # Prepare the input data
    input_data = np.array([brand, model_input, variant, name, year, km_driven, fuel, seller_type, transmission, owner])

    # Separate numerical fields from non-numerical ones
    numerical_data = np.array([year, km_driven])  # Only numerical fields (year, km_driven in this case)

    # Check for missing or NaN values in the numerical fields
    if np.isnan(numerical_data).any():
        st.error("Numerical inputs contain missing values. Please correct the input and try again.")
    elif any(x == '' for x in [brand, model_input, variant, name, fuel, seller_type, transmission, owner]):
        st.error("Please fill in all the categorical details.")
    else:
        # Reshape input if necessary (assuming the model expects a 2D array)
        input_data = input_data.reshape(1, -1)  # 1 sample, multiple features
        
        # Perform the prediction using the loaded model
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"The predicted selling price is: {predicted_price:.2f}")
        except AttributeError as e:
            st.error(f"An attribute error occurred during prediction: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Saving model
# Set up the Random Forest model
final_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Train the Random Forest model
# final_model.fit(x, y)  # Uncomment when ready to retrain the model with your data

# Save the final model using pickle
model_directory = "../models"

# Check if the directory exists, if not, create it
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save the final model using pickle
try:
    pickle.dump(final_model, open(f"{model_directory}/Random_Forest.pkl", "wb"))
    st.success(f"Model saved successfully at: {model_directory}/Random_Forest.pkl")
except Exception as e:
    st.error(f"An error occurred while saving the model: {e}")

# Load the model
try:
    load_model = pickle.load(open(f"{model_directory}/Random_Forest.pkl", "rb"))
    st.success(f"Model loaded successfully: Random Forest")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

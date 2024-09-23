# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import warnings
warnings.filterwarnings("ignore")
import gzip
import numpy as np

# Load your saved Random Forest model
with gzip.open('Random_Forest_compressed.pkl', 'rb') as file:
    model = joblib.load(file)

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
    # Check if any required fields are missing
    if not all([brand, model_input, variant, name, year, km_driven, fuel, seller_type, transmission, owner]):
        st.error("Please fill in all the details.")
    else:
        # Prepare the input data in the correct format for the model
        input_data = np.array([[brand, model_input, variant, name, year, km_driven, fuel, seller_type, transmission, owner]])

        # Reshape if necessary (assuming the model expects a 2D array)
        input_data = input_data.reshape(1, -1)  # 1 sample, multiple features
        
        # Check if model supports missing values (if any NaNs are present)
        if np.isnan(input_data).any():
            st.error("Input data contains missing values. Please correct the input and try again.")
        else:
            # Perform the prediction using the loaded model
            try:
                predicted_price = model.predict(input_data)[0]
                st.success(f"The predicted selling price is: {predicted_price:.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

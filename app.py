import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the directory of the current script
current_directory = os.path.dirname(__file__)

# Construct the full path to the model file
model_path = os.path.join(current_directory, 'Gradient_boosting_model.pkl')

# Load the trained Gradient Boosting model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'Gradient_boosting_model.pkl' is in the correct directory.")
    st.stop()  # Stop the app if the model cannot be loaded

# Load the CSV file using pandas
df = pd.read_csv('cleaned_data.csv')

# Extract the column names from the DataFrame
expected_columns = df.columns.tolist()

# Save the expected columns to a file 
joblib.dump(expected_columns, 'expected_columns.pkl')

st.title("Car Selling Price Prediction App")
st.write("### Predict the selling price of a car based on its details.")

# Collect user input
brand = st.selectbox('Select Car Brand', ['Toyota', 'Honda', 'Ford', 'Other'])
model_input = st.text_input('Enter Model')
variant = st.text_input('Enter Variant')
name = st.text_input('Enter Name')
year = st.number_input('Enter the Year of Manufacture', min_value=1990, max_value=2024, step=1)
km_driven = st.number_input('Enter Km Driven', min_value=0)
fuel = st.selectbox('Select Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric'])
seller_type = st.selectbox('Select Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Select Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Select Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above'])

# Prepare the input data as a dictionary
input_dict = {
    'brand': brand,
    'model': model_input,
    'variant': variant,
    'name': name,
    'year': year,
    'km_driven': km_driven,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner
}

# Convert input into a DataFrame
input_data = pd.DataFrame([input_dict])

# Apply one-hot encoding to categorical features (must match the training encoding)
input_data_encoded = pd.get_dummies(input_data, columns=['brand', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Ensure the input_data_encoded has the same columns as expected_columns
for col in expected_columns:
    if col not in input_data_encoded:
        input_data_encoded[col] = 0

# Reorder columns to match the training data
input_data_encoded = input_data_encoded[expected_columns]

# Perform prediction
if st.button('Predict Selling Price'):
    predicted_price = model.predict(input_data_encoded)[0]
    st.success(f"The predicted selling price is: {predicted_price:.2f}")

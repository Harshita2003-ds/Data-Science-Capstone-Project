import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import numpy as np

# Load your saved Gradient Boosting model
try:
    model = joblib.load('Gradient_boosting_model.pkl')
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Define your app's title and description
st.title("Car Selling Price Prediction App")
st.write("### Predict the selling price of a car based on its details.")

# Create input fields for user to enter data
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

        # Load the model's expected feature names from the fitted model
        expected_features = model.feature_names_in_  # Feature names used during fitting

        # Ensure the input DataFrame has the same columns as the model expects
        input_data_encoded = input_data_encoded.reindex(columns=expected_features, fill_value=0)

        # Perform the prediction using the loaded model
        try:
            predicted_price = model.predict(input_data_encoded)[0]
            st.success(f"The predicted selling price is: {predicted_price:.2f}")
        except AttributeError as e:
            st.error(f"An attribute error occurred during prediction: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

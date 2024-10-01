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

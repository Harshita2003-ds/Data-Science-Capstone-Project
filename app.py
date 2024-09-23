# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import warnings
warnings.filterwarnings("ignore")
import gzip


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
km_driven = st.number_input('Enter Km Driven')
fuel = st.selectbox('Select Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric'])  # Add more fuel types
seller_type = st.selectbox('Select Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Select Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Select Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above'])

# Button to trigger prediction
if st.button('Predict Selling Price'):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[brand, model_input, variant, name, year, km_driven, fuel, seller_type, transmission, owner]],
                              columns=['Brand', 'Model', 'Variant', 'Name', 'Year', 'Km_Driven', 'Fuel', 'Seller_Type', 'Transmission', 'Owner'])
    
    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    # Display the result
    st.success(f"The predicted selling price for the car is ₹{predicted_price:,.2f}")

# Run the app using 'streamlit run app.py'

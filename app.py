# app.py
import streamlit as st
import numpy as np
from tensorflow import keras

# Load the saved model weights
model = keras.models.load_model(r'C:\Users\Vashistha.000\Desktop\DDOS\imbal_dnn_model.h5')

# Function to preprocess input data for prediction
def preprocess_data(input_data):
    # Perform any necessary preprocessing steps here
    # For example, scaling, encoding categorical variables, etc.
    preprocessed_data = input_data  # Placeholder for now
    return preprocessed_data

# Function to make predictions using the loaded model
def predict_ddos(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data)
    # Make predictions
    predictions = model.predict(preprocessed_data)
    return predictions

# Streamlit app
def main():
    st.title('DDoS Detection Model')

    # Input form for users to provide data
    st.header('Input Data')
    # Example input fields, you can customize these according to your dataset
    src_ip = st.text_input('Source IP')
    src_port = st.number_input('Source Port', min_value=0, max_value=65535)
    # Add more input fields as needed...

    # Button to trigger predictions
    if st.button('Predict'):
        # Collect input data
        input_data = np.array([[src_ip, src_port]])  # Example, you'll need to format this according to your preprocessing function
        # Make predictions
        predictions = predict_ddos(input_data)
        # Display predictions
        st.header('Prediction Results')
        st.write(predictions)  # Display predictions however you want

    # Show the values inside X_test
    st.subheader('Values inside X_test')
    st.write(X_test)

if __name__ == '__main__':
    main()

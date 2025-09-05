import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load(r'C:\Users\jaraneses\OneDrive - 2X LLC\Codes\Restaurant Predictor\Scaler.pkl')

st.set_page_config(layout= 'wide')

st.title('Restaurant Rating Prediction App')

st.caption('This app helps you to predict a restaurant review class')

st.divider()
    
averagecost = st.number_input('Please enter the estimated the average cost for two', min_value = 15, max_value = 999999999999, value = 1000, step = 200)

tablebooking = st.selectbox('Restaurant has table booking?', ['Yes', 'No'])

onlinedelivery = st.selectbox('Restaurant has online booking?',['Yes', 'No'])

pricerange = st.selectbox('What is the price range (1 Cheapest, 4 Most Expensive)',[1, 2, 3, 4])

predictbutton = st.button('Predict the review')

model = joblib.load(r'C:\Users\jaraneses\OneDrive - 2X LLC\Codes\Restaurant Predictor\mlmodel.pkl')

bookingstatus = 1 if tablebooking == 'Yes' else 0

deliverystatus = 1 if onlinedelivery == ' Yes' else 0

if predictbutton:
    # Create input array properly
    my_X_values = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])  # Double brackets for 2D
    
    # Transform and predict
    X = scaler.transform(my_X_values)
    prediction = model.predict(X)

    if prediction < 2.5:
        st.write('Poor')
    elif prediction < 3.5:
        st.write('Average')
    elif prediction < 4.0:
        st.write('Good')
    elif prediction < 4.5:
        st.write ('Very Good')
    else:
        st.write('Excellent')
    
    # Display results
    st.snow()
    st.success(f"Predicted Restaurant Rating: {prediction[0]:.2f}")
    




# Load the trained Lasso regression model
import pickle
import pandas as pd
import streamlit as st

filename = 'lasso_regression_model.sav'
model = pickle.load(open(filename, 'rb'))

st.title("Monthly Revenue Prediction")

# Input fields for user data
st.header("Enter Store Data")
no_of_orders = st.number_input("Number of orders", min_value=0)
avg_order_value = st.number_input("Average Order Value", min_value=0.0)
total_customers = st.number_input("Total Customers", min_value=0)
customer_acquisition_cost = st.number_input("Customer Acquisition Cost", min_value=0.0)
avg_customer_lifetime_value = st.number_input("Average Customer Lifetime Value", min_value=0.0)
operational_costs = st.number_input("Operational Costs", min_value=0.0)
website_traffic = st.number_input("Website Traffic", min_value=0)
conversion_rate = st.number_input("Conversion Rate", min_value=0.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0)
social_media_engagement = st.number_input("Social Media Engagement", min_value=0)
customer_satisfaction = st.number_input("Customer Satisfaction", min_value=0.0)
customer_retention_rate = st.number_input("Customer Retention Rate", min_value=0.0)

# Create a button to make predictions
if st.button("Predict"):
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'no_of_orders': [no_of_orders],
        'avg_order_value': [avg_order_value],
        'total_customers': [total_customers],
        'customer_acquisition_cost': [customer_acquisition_cost],
        'avg_customer_lifetime_value': [avg_customer_lifetime_value],
        'operational_costs': [operational_costs],
        'website_traffic': [website_traffic],
        'conversion_rate': [conversion_rate],
        'marketing_spend': [marketing_spend],
        'social_media_engagement': [social_media_engagement],
        'customer_satisfaction': [customer_satisfaction],
        'customer_retention_rate': [customer_retention_rate],
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Display the prediction
    st.header("Prediction:")
    st.write(f"The predicted monthly revenue is: {prediction:.2f}")

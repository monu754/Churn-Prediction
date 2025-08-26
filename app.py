import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
pipeline = joblib.load('saved_model/churn_model_pipeline.joblib')

# --- STREAMLIT APP ---

st.title('Customer Churn Prediction 🔮')
st.write("Is this customer at risk of leaving? Use our AI-powered tool to see their churn probability. Just fill in their details to get started.")

# Create input fields for user to enter customer data
# We need to create inputs for all the features the model was trained on.

st.header("Customer Details")

# Organize inputs into columns for better layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])

with col2:
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])

st.header("Contract and Payment Details")

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

with col4:
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=500.0)


# Create a button to make a prediction
if st.button('Predict Churn', type="primary"):
    # Create a DataFrame from the user inputs in the correct order
    input_data = pd.DataFrame({
        'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner], 'Dependents': [dependents],
        'tenure': [tenure], 'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
        'InternetService': [internet_service], 'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection], 'TechSupport': [tech_support], 'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies], 'Contract': [contract], 'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
    })
    
    # Get prediction probability
    churn_probability = pipeline.predict_proba(input_data)[0][1]
    
    # Display the result
    st.subheader('Prediction Result')
    if churn_probability > 0.5:
        st.error(f'High Risk of Churn (Probability: {churn_probability:.2%})')
        st.write("Recommendation: Proactively engage this customer with retention offers or support.")
    else:
        st.success(f'Low Risk of Churn (Probability: {churn_probability:.2%})')
        st.write("Recommendation: Continue to provide excellent service to maintain loyalty.")
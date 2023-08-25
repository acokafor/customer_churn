import streamlit as st
import pickle
import pandas as pd

# Load the saved SVM model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess user inputs and make predictions
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

def main():
    st.title('Customer Churn Prediction App')

    st.sidebar.header('User Input Features')

    # Collect user input using Streamlit components
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 1)
    monthly_charges = st.sidebar.slider('Monthly Charges', 18, 120, 50)
    total_charges = st.sidebar.slider('Total Charges', 18, 8000, 2000)

    gender = st.sidebar.radio('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'])

    # Create a dictionary from the user inputs
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Perform one-hot encoding for categorical features
    categorical_columns = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    encoded_input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    # Make predictions
    if st.button('Predict'):
        prediction = predict_churn(encoded_input_df)
        st.write('Prediction:', prediction[0])

if __name__ == '__main__':
    main()
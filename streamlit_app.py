import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load the trained model from the pickle file
model_file = 'model.pkl'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# UI and Prediction Logic
def main():
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')
    st.image(image, use_column_width=False)
    
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    
    st.title("Predicting Customer Churn")
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
    
    if add_selectbox == 'Online':
        st.sidebar.title("Online Prediction")
        
        # UI components for user input
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phoneservice = st.selectbox('Customer has phoneservice:', ['yes', 'no'])
        multiplelines = st.selectbox('Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
        internetservice = st.selectbox('Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity = st.selectbox('Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox('Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox('Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox('Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox('Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox('Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox('Customer has paperlessbilling:', ['yes', 'no'])
        paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
        
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges

        if st.button("Predict"):
            input_dict = {
                "gender": gender,
                "seniorcitizen": seniorcitizen,
                "partner": partner,
                "dependents": dependents,
                "phoneservice": phoneservice,
                "multiplelines": multiplelines,
                "internetservice": internetservice,
                "onlinesecurity": onlinesecurity,
                "onlinebackup": onlinebackup,
                "deviceprotection": deviceprotection,
                "techsupport": techsupport,
                "streamingtv": streamingtv,
                "streamingmovies": streamingmovies,
                "contract": contract,
                "paperlessbilling": paperlessbilling,
                "paymentmethod": paymentmethod,
                "tenure": tenure,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges
            }

            # Preprocess input_dict if needed (aligning with training preprocessing)
            # Make sure the input_dict keys match the features expected by your model
            
            X = pd.DataFrame.from_dict([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
    
    elif add_selectbox == 'Batch':
        st.sidebar.title("Batch Prediction")
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            # Preprocess 'data' DataFrame if needed (aligning with training preprocessing)
            X = data  # Assuming the data is already preprocessed
            y_preds = model.predict_proba(X)[:, 1]
            churn_results = y_preds >= 0.5
            st.write("Churn predictions:")
            st.write(churn_results)

if __name__ == '__main__':
    main()
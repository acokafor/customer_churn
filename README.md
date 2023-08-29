# Customer Churn Prediction Streamlit App

This is a Streamlit web application for predicting customer churn based on a trained Support Vector Machine (SVM) model. The app allows users to input customer information and get predictions about whether a customer is likely to churn or stay.

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/yourusername/customer-churn-streamlit-app.git
   ```

2. Navigate to the project directory:
   ```
   cd customer-churn-streamlit-app
   ```

3. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. After installing the required packages, you can launch the Streamlit app using the following command:
   ```
   streamlit run streamlit_app.py
   ```

2. This will start a local development server and open the app in your default web browser.

## Viewing the App

1. Once the app is running, you will see a form with input fields for customer features.
2. Enter values for each feature to make a prediction.
3. Click the "Predict" button to see the prediction result displayed on the page.


## Acknowledgments

- The dataset used for training the SVM model is from [Dataset Source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- The SVM model was trained using scikit-learn [Scikit-learn](https://scikit-learn.org/).

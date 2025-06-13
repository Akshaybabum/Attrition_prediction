import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Load data and preprocess
@st.cache_data
def load_data():
    data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    categorical_cols.remove('Attrition')
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    return data

# Load and preprocess data
data = load_data()

# Train/Test Split
from sklearn.model_selection import train_test_split
X = data.drop('Attrition', axis=1)
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (or load from file if you already trained it)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

st.title("HR Employee Attrition Predictor")

# User input form
st.sidebar.header("Enter Employee Details")

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]
st.write("## Prediction:")
st.success("Likely to Stay" if prediction == 0 else "Likely to Leave")

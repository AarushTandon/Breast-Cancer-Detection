import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", 
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", 
    "radius error", "texture error", "perimeter error", "area error", "smoothness error", 
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error", 
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", 
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Prediction function
def predict_cancer(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "✅ The Breast Cancer is Benign"
    else:
        return "⚠️ The Breast Cancer is Malignant"

# Streamlit UI
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter patient’s test values below to predict if the tumor is **Benign** or **Malignant**.")

# Sidebar inputs
st.sidebar.header("Enter Features")
user_inputs = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", value=0.0, format="%.5f")
    user_inputs.append(value)

# Predict button
if st.sidebar.button("Predict"):
    result = predict_cancer(user_inputs)
    st.success(result)

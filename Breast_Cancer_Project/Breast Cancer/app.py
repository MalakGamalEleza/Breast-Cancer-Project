import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved files
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

st.title("Breast Cancer Survival Prediction")

st.write(
    "Predict Overall Survival Status "
    "using clinical features."
)

# User Inputs
age = st.number_input(
    "Age at Diagnosis",
    min_value=0.0,
    value=50.0
)

tumor_size = st.number_input(
    "Tumor Size",
    min_value=0.0,
    value=20.0
)

tumor_stage = st.number_input(
    "Tumor Stage",
    min_value=0.0,
    value=2.0
)

mutation_count = st.number_input(
    "Mutation Count",
    min_value=0.0,
    value=5.0
)

neoplasm_histologic_grade = st.number_input(
    "Neoplasm Histologic Grade",
    min_value=0.0,
    value=2.0
)

lymph_nodes_examined_positive = st.number_input(
    "Lymph Nodes Examined Positive",
    min_value=0.0,
    value=1.0
)

# Feature Engineering
tumor_size_per_stage = (
    tumor_size /
    (tumor_stage + 1)
)

# Create input dataframe
input_data = pd.DataFrame([{
    'Age at Diagnosis': age,
    'Tumor Size': tumor_size,
    'Tumor Stage': tumor_stage,
    'Mutation Count': mutation_count,
    'Neoplasm Histologic Grade':
    neoplasm_histologic_grade,
    'Lymph nodes examined positive':
    lymph_nodes_examined_positive,
    'Tumor_Size_per_Stage':
    tumor_size_per_stage
}])

# Add missing columns
training_columns = selector.feature_names_in_

for col in training_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Correct order
input_data = input_data[
    training_columns
]

# Scale data
scaled_data = scaler.transform(
    input_data
)

# Feature selection
selected_data = selector.transform(
    scaled_data
)

# Prediction button
if st.button("Predict"):

    prediction = model.predict(
        selected_data
    )[0]

    if prediction == 1:
        st.success(
            "Patient likely survived"
        )
    else:
        st.error(
            "Patient likely not survived"
        )
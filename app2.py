import streamlit as st
import pandas as pd
import joblib

st.title("Insurance Premium Prediction")

with open('./Model/insurance_model.pkl', 'rb') as file:
    model = joblib.load(file)
  
if model is not None:
    col1, col2 = st.columns(2)
            
    with col1:
        age = st.number_input("Age", 18, 100)
            
    with col2:
        bmi = st.number_input("BMI", 18.00, 53.10)

    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
            
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", ['Male', 'Female'])
            
    with col4:
        is_smoker = st.selectbox("Smoker", ['No', 'Yes'])

    region = st.selectbox("Region", ['Southwest', 'Southeast', 'Northwest', 'Northeast'])

    def encode_region(region_name):
        region_map = {
            "Southwest": 0,
            "Southeast": 1,
            "Northwest": 2,
            "Northeast": 3
        }
        return region_map.get(region_name, 0)

    age = float(age)
    gender = 1 if gender == "Male" else 0
    bmi = float(bmi)
    children = int(children)
    is_smoker = 1 if is_smoker == "Yes" else 0
    region_encoded = encode_region(region)

    if st.button('Predict it', type='primary', use_container_width=True):
        try:
            input_data = pd.DataFrame({
                'age': [age],
                'bmi': [bmi],
                'children': [children],
                'gender_label_encoded': [gender],
                'region_label_encoded': [region_encoded],
                'smoker_label_encoded': [is_smoker],
            })
            
            input_data = input_data.astype(float)

            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Insurance Premium: ${prediction:,.2f}")
            
        except Exception as e:
            st.error(f"Error: {e}")

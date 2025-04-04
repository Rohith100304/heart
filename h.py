import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import base64
import io

# Load the saved model and dataset
@st.cache_resource
def load_heart_model():
    return load_model('heart')

@st.cache_data
def load_dataset():
    return pd.read_csv('heart.csv')

model = load_heart_model()
data = load_dataset()

# File download functions
def download_dataset():
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def download_model():
    with open('heart.pkl', 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'data:file/pkl;base64,{b64}'
    return href

# Create a function to get user input in main area
def get_user_input():
    st.header("Patient Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=70, max_value=220, value=150)
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0)
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', 
                           ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    with col2:
        # Categorical inputs
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox('Resting Electrocardiographic Results', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                            ['Upsloping', 'Flat', 'Downsloping'])
        
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_mapping[cp]
    fbs = 1 if fbs == 'Yes' else 0
    restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == 'Yes' else 0
    slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 0}
    slope = slope_mapping[slope]
    thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = thal_mapping[thal]
    
    # Store a dictionary into a dataframe
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    # Title
    st.title('Heart Disease Prediction App')
    st.write("""
    This app predicts the likelihood of having heart disease based on patient information.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # View Dataset button
    if st.sidebar.button("View Dataset"):
        st.subheader("Heart Disease Dataset")
        st.write(data)
    
    # Download Dataset button
    dataset_download = download_dataset()
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='heart_dataset.csv',
        mime='text/csv'
    )
    
    # Download Model button
    with open('heart.pkl', 'rb') as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Download Model",
        data=model_bytes,
        file_name='heart_model.pkl',
        mime='application/octet-stream'
    )
    
    # Get user input in main area
    user_input = get_user_input()
    
    # Display user input
    st.subheader('Patient Input Summary')
    st.write(user_input)
    
    # Prediction button
    if st.button('Predict Heart Disease Risk'):
        # Make prediction
        prediction = predict_model(model, data=user_input)
        
        # Display prediction
        st.subheader('Prediction Result')
        prediction_value = prediction['prediction_label'][0]
        prediction_score = prediction['prediction_score'][0]
        
        if prediction_value == 1:
            st.error(f'**High risk of heart disease** (Probability: {prediction_score:.2%})')
            st.warning('Please consult with a healthcare professional for further evaluation.')
        else:
            st.success(f'**Low risk of heart disease** (Probability: {1 - prediction_score:.2%})')
            st.info('Maintain a healthy lifestyle for continued heart health.')
            
        # Show prediction details expander
        with st.expander("Show detailed prediction metrics"):
            st.write(prediction)

if __name__ == '__main__':
    main()

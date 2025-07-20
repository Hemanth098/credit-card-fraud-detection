import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load models
try:
    with open('Models/CNN_model.pkl', 'rb') as f:
        cnn_model = pickle.load(f)
    with open('Models/LSTM_model.pkl', 'rb') as f:
        lstm_model = pickle.load(f)
    with open('Models/SVM_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    cnn_lstm = load_model('Models/new.h5')
    with open('Models/Logreg_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Ensure session state for data storage
if 'data' not in st.session_state:
    columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    st.session_state.data = pd.DataFrame(columns=columns)

# Sidebar inputs
st.sidebar.header("Update Features")
input_data = {
    'Time': st.sidebar.number_input("Time", value=0.0, step=1.0),
    'Amount': st.sidebar.number_input("Amount", value=0.0, step=0.1)
}
input_data.update({f'V{i}': st.sidebar.number_input(f'V{i}', value=0.0, step=0.1) for i in range(1, 29)})

# Button to update DataFrame
if st.sidebar.button("Update Data"):
    new_data = pd.DataFrame([input_data])
    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

# Text area for direct CSV input
st.sidebar.header("Add Data via CSV Input")
user_input = st.sidebar.text_area("Enter CSV data (comma-separated)")
if st.sidebar.button("Add CSV Data"):
    try:
        new_row = pd.read_csv(pd.io.common.StringIO(user_input), header=None)
        new_row.columns = st.session_state.data.columns  # Ensure column names match
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        st.success("Data added successfully!")
    except Exception as e:
        st.error(f"Invalid input format: {e}")
st.write("Expected columns:", len(st.session_state.data.columns))

# Display DataFrame on the main page
st.title("Updated DataFrame")
st.dataframe(st.session_state.data)

# Prediction function
def make_prediction(model, deep_learning=False, lstm=False, svm=False):
    if not st.session_state.data.empty:
        input_values = st.session_state.data.iloc[-1:].values
        input_values = np.array(input_values).astype(np.float32)  # Ensure proper format

        if lstm:
            input_values = input_values[:, 2:11]  # Select only the first 9 PCA features (excluding Time and Amount)
            input_values = input_values.reshape(1, 1, 9)  # Reshape for LSTM
        elif deep_learning:
            input_values = input_values.reshape(1, -1)  # Reshape for CNN or CNN-LSTM
        elif svm:
            input_values = input_values[:, 2:30]  # Use only PCA features (V1-V28)

        prediction = model.predict(input_values)
        rounded_prediction = int(round(prediction[0][0]))  # Round to nearest integer

        if rounded_prediction == 0:
            st.markdown("<p style='color:green; font-size:20px;'>Non-fraudulent</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:red; font-size:20px;'>Fraudulent</p>", unsafe_allow_html=True)
    else:
        return "No data available"

# Prediction buttons
if st.button("Predict with CNN"):
    make_prediction(cnn_model, deep_learning=True)

if st.button("Predict with LSTM"):
    make_prediction(lstm_model, deep_learning=True, lstm=True)


if st.button("Predict with CNN-LSTM"):
    make_prediction(cnn_lstm, deep_learning=True)


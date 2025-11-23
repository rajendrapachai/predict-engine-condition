
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID_MODEL = "RajendrakumarPachaiappan/engine-predictive-model"

MODEL_FILENAME = "final_random_forest_model.joblib"
SCALER_FILENAME = "standard_scaler.joblib"

# Feature Column for Input
FEATURE_COLS = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'
]

# Caching Function to Load Model and Scaler
@st.cache_resource
def load_artifacts():
    try:
        with st.spinner("Downloading and loading model artifacts..."):
            # Download and load the model
            model_path = hf_hub_download(repo_id=REPO_ID_MODEL, filename=MODEL_FILENAME, repo_type="model")
            model = joblib.load(model_path)

            # Download and load the scaler
            scaler_path = hf_hub_download(repo_id=REPO_ID_MODEL, filename=SCALER_FILENAME, repo_type="model")
            scaler = joblib.load(scaler_path)

        return model, scaler
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load model or scaler. Check repository ID and filenames. Error: {e}")
        return None, None

# Load the model and scaler
model, scaler = load_artifacts()

# Streamlit Setup
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Predict Engine Condition")
st.markdown(
    "This model forecasts potential engine failures, classifying the status as **Normal (0)** or requiring **Immediate Maintenance (1)**."
)
st.markdown(
    "**Note:** Adjust the sliders below with current sensor readings to check the engine condition."
)

if model is None or scaler is None:
    st.stop()

st.header("Sensor Readings")

ranges = {
    'Engine_RPM': (61.0, 2239.0, 791.0),
    'Lub_Oil_Pressure': (0.0, 7.3, 3.3),
    'Fuel_Pressure': (0.0, 21.1, 6.7),
    'Coolant_Pressure': (0.0, 7.5, 2.3),
    'Lub_Oil_Temperature': (71.3, 89.6, 77.6),
    'Coolant_Temperature': (61.7, 195.5, 78.4),
}

input_values = {}
col1, col2 = st.columns(2)
columns = [col1, col2]

for i, col_name in enumerate(FEATURE_COLS):
    current_col = columns[i % 2]

    min_val, max_val, default_val = ranges[col_name]

    label = col_name.replace('_', ' ')
    unit = ""
    if "RPM" in col_name:
        unit = " (rev/min)"
    elif "Pressure" in col_name:
        unit = " (bar/kPa)"
    elif "Temperature" in col_name:
        unit = " (°C)"


    with current_col:
        input_values[col_name] = st.slider(
            label=f"{label}{unit}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1,
            help=f"Current reading for {label}. Full data range: [{min_val}, {max_val}]"
        )

# Prediction Logic
if st.button("Predict Engine Condition", type="primary"):
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLS)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "FAULTY (1): Immediate Maintenance Required! "
            "High probability of engine failure detected. Check for high RPM, low pressures, or extreme temperatures."
        )
    else:
        st.success(
            "NORMAL (0): Operating within expected parameters. "
            "Engine health is currently good."
        )

    st.caption(f"Raw Model Prediction (0=Normal, 1=Faulty): {prediction}")

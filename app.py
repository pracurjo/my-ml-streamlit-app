import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# UI FIX: Bigger Question, Smaller Option
# =========================
st.markdown("""
<style>
/* Question / label font bigger */
label, .stSlider label, .stSelectbox label {
    font-size: 1.15rem !important;
    font-weight: 600;
}

/* Selected option text smaller */
div[data-baseweb="select"] span {
    font-size: 0.9rem !important;
}

/* Slider value text smaller */
.stSlider > div > div > div > div {
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model Files
# =========================
@st.cache_resource
def load_files():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_files()

# =========================
# App Header
# =========================
st.title("❤️ Heart Disease Risk Prediction System")
st.write(
    "This application estimates the **risk of heart disease (%)** "
    "based on basic health information provided by the user."
)

st.markdown("---")

# =========================
# User Inputs (FORMAT UNCHANGED)
# =========================
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina (chest pain during physical activity)",
        "Atypical Angina (unusual chest pain)",
        "Non-Anginal Pain (not heart related)",
        "Asymptomatic (no chest pain)"
    ]
)

resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 600, 200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])

resting_ecg = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T Abnormality (ECG changes)",
        "Left Ventricular Hypertrophy (thickened heart wall)"
    ]
)

max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exercise_angina = st.selectbox("Chest Pain During Exercise?", ["No", "Yes"])

oldpeak = st.slider("Oldpeak (ST Depression during exercise)", 0.0, 6.0, 1.0)

st_slope = st.selectbox(
    "ST Segment Slope",
    [
        "Upward (usually normal)",
        "Flat (moderate risk)",
        "Downward (higher risk)"
    ]
)

# =========================
# Prediction
# =========================
st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    chest_map = {
        "Typical Angina (chest pain during physical activity)": "TA",
        "Atypical Angina (unusual chest pain)": "ATA",
        "Non-Anginal Pain (not heart related)": "NAP",
        "Asymptomatic (no chest pain)": "ASY"
    }

    ecg_map = {
        "Normal": "Normal",
        "ST-T Abnormality (ECG changes)": "ST",
        "Left Ventricular Hypertrophy (thickened heart wall)": "LVH"
    }

    slope_map = {
        "Upward (usually normal)": "Up",
        "Flat (moderate risk)": "Flat",
        "Downward (higher risk)": "Down"
    }

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        f"ChestPainType_{chest_map[chest_pain]}": 1,
        f"RestingECG_{ecg_map[resting_ecg]}": 1,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        f"ST_Slope_{slope_map[st_slope]}": 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if hasattr(model, "predict_proba"):
        risk_percent = model.predict_proba(scaled_input)[0][1] * 100
    else:
        risk_percent = 100 if prediction == 1 else 0

    # =========================
    # Output
    # =========================
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"📊 **Estimated Risk:** {risk_percent:.1f}%")
    st.progress(int(risk_percent))

    st.write("🩺 **Health Suggestions**")

    reason_found = False

    if age > 55:
        st.write("• Higher age increases the risk of heart disease.")
        reason_found = True
    if cholesterol > 240:
        st.write("• High cholesterol may lead to blocked blood vessels.")
        reason_found = True
    if fasting_bs == "Yes":
        st.write("• High blood sugar increases cardiovascular risk.")
        reason_found = True
    if exercise_angina == "Yes":
        st.write("• Chest pain during exercise is an important warning sign.")
        reason_found = True
    if oldpeak > 2:
        st.write("• Significant ECG changes detected during exercise.")
        reason_found = True

    if not reason_found:
        st.write("• No major high-risk factors were detected from the provided information.")

    st.markdown("---")
    st.info(
        "This is an AI-based risk estimation tool and does not replace professional medical advice."
    )

# =========================
# Footer
# =========================
st.caption("Developed by Pracurjo | AI Health Prediction Project")

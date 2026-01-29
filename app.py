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
st.markdown("""
এই অ্যাপটি আপনার স্বাস্থ্য সংক্রান্ত তথ্য ব্যবহার করে  
**হার্ট ডিজিজ হওয়ার ঝুঁকি (%)** সহজ ভাষায় দেখায়।  
ℹ️ ইনপুট ফিল্ডের পাশে থাকা **(?)** আইকনে মাউস রাখলে সংশ্লিষ্ট তথ্যের ব্যাখ্যা প্রদর্শিত হবে।


""")

st.markdown("---")

# =========================
# User Inputs (All with HELP)
# =========================
age = st.slider(
    "Age (বয়স)",
    18, 100, 40,
    help="আপনার বর্তমান বয়স নির্বাচন করুন"
)

sex = st.selectbox(
    "Sex (লিঙ্গ)",
    ["Male", "Female"],
    help="Male = পুরুষ, Female = নারী"
)

chest_pain = st.selectbox(
    "Chest Pain Type (বুকের ব্যথার ধরন)",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic (কোনো ব্যথা নেই)"
    ],
    help="""
Typical Angina: পরিশ্রম করলে বুকের মাঝখানে চাপ  
Atypical Angina: অস্বাভাবিক বুক ব্যথা  
Non-Anginal Pain: হার্টজনিত নয়  
Asymptomatic: কোনো বুক ব্যথা নেই
"""
)

resting_bp = st.slider(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120,
    help="বিশ্রাম অবস্থায় রক্তচাপ (সাধারণত 120/80)"
)

cholesterol = st.slider(
    "Cholesterol Level (mg/dL)",
    100, 600, 200,
    help="রক্তে কোলেস্টেরল (২০০ এর নিচে হলে ভালো)"
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL?",
    ["No", "Yes"],
    help="না খেয়ে রক্ত পরীক্ষা করলে সুগার 120 এর বেশি হলে Yes"
)

resting_ecg = st.selectbox(
    "Resting ECG Result",
    ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
    help="ECG পরীক্ষার ফলাফল"
)

max_hr = st.slider(
    "Maximum Heart Rate Achieved",
    60, 220, 150,
    help="ব্যায়াম বা হাঁটার সময় সর্বোচ্চ হার্ট রেট"
)

exercise_angina = st.selectbox(
    "Chest Pain During Exercise?",
    ["No", "Yes"],
    help="ব্যায়ামের সময় বুক ব্যথা হলে Yes নির্বাচন করুন"
)

oldpeak = st.slider(
    "Oldpeak (ST Depression Level)",
    0.0, 6.0, 1.0,
    help="Exercise সময় ECG তে ST segment কতটা নিচে নামে"
)

st_slope = st.selectbox(
    "ST Segment Slope",
    ["Upward", "Flat", "Downward"],
    help="""
Upward: সাধারণত স্বাভাবিক  
Flat: মাঝারি ঝুঁকি  
Downward: হার্ট সমস্যার ঝুঁকি বেশি
"""
)

# =========================
# Prediction
# =========================
st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    # ---------- Mapping ----------
    chest_map = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP",
        "Asymptomatic (কোনো ব্যথা নেই)": "ASY"
    }

    ecg_map = {
        "Normal": "Normal",
        "ST-T Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH"
    }

    slope_map = {
        "Upward": "Up",
        "Flat": "Flat",
        "Downward": "Down"
    }

    # ---------- Raw Input ----------
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

    # ---------- Prediction ----------
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

    st.subheader(f"📊 Estimated Risk: {risk_percent:.1f}%")
    st.progress(int(risk_percent))

    st.subheader("🩺 Health Suggestions")
    reason_found = False

    if age > 55:
        st.write("🔸 বয়স বেশি হলে হার্ট ডিজিজের ঝুঁকি বাড়ে")
        reason_found = True
    if cholesterol > 240:
        st.write("🔸 কোলেস্টেরল বেশি হলে রক্তনালী ব্লক হওয়ার ঝুঁকি")
        reason_found = True
    if fasting_bs == "Yes":
        st.write("🔸 রক্তে সুগার বেশি হলে হার্ট ঝুঁকি বাড়ে")
        reason_found = True
    if exercise_angina == "Yes":
        st.write("🔸 ব্যায়ামের সময় বুক ব্যথা গুরুত্বপূর্ণ লক্ষণ")
        reason_found = True
    if oldpeak > 2:
        st.write("🔸 ECG তে বেশি ST depression দেখা গেছে")
        reason_found = True

    if not reason_found:
        st.write("✅ আপনার দেওয়া তথ্য অনুযায়ী উল্লেখযোগ্য কোনো বড় ঝুঁকির কারণ ধরা পড়েনি।")

    st.markdown("---")
    st.info("⚠️ এটি একটি AI-based prediction। চূড়ান্ত সিদ্ধান্তের জন্য অবশ্যই ডাক্তারের পরামর্শ নিন।")

# =========================
# Footer
# =========================
st.caption("Developed by Pracurjo | AI Health Prediction Project")

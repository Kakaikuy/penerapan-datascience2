import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# LOAD MODEL & PIPELINE
# ===============================
MODEL_PATH = r"C:\Users\ASUS\Documents\Python project\Penerapan Data Science P2\model"

model = joblib.load(os.path.join(MODEL_PATH, "logreg_model.joblib"))  # atau knn_model, tree_model
target_encoder = joblib.load(os.path.join(MODEL_PATH, "encoder_target.joblib"))
pca_1 = joblib.load(os.path.join(MODEL_PATH, "pca_1.joblib"))
pca_2 = joblib.load(os.path.join(MODEL_PATH, "pca_2.joblib"))

numerical_pca_1 = [
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]

numerical_pca_2 = [
    'Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

categorical_columns = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Displaced", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder"
]

# Load encoders & scalers
scalers = {col: joblib.load(os.path.join(MODEL_PATH, f"scaler_{col}.joblib")) for col in numerical_pca_1 + numerical_pca_2}
encoders = {col: joblib.load(os.path.join(MODEL_PATH, f"encoder_{col}.joblib")) for col in categorical_columns}

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Isi form di bawah ini untuk memprediksi status mahasiswa berdasarkan data pendaftaran dan akademik.")

with st.form("student_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        for col in categorical_columns[:len(categorical_columns)//2]:
            inputs[col] = st.selectbox(col.replace("_", " ").capitalize(), list(encoders[col].categories_[0]))

        for col in numerical_pca_1[:4]:
            inputs[col] = st.number_input(col.replace("_", " ").capitalize(), step=0.1, format="%.2f")

    with col2:
        for col in categorical_columns[len(categorical_columns)//2:]:
            inputs[col] = st.selectbox(col.replace("_", " ").capitalize(), list(encoders[col].categories_[0]))

        for col in numerical_pca_1[4:] + numerical_pca_2:
            inputs[col] = st.number_input(col.replace("_", " ").capitalize(), step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 Prediksi Status Mahasiswa")

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_df = pd.DataFrame([inputs])

    # Transformasi kategori
    for col in categorical_columns:
        input_df[[col]] = encoders[col].transform(input_df[[col]])

    # Transformasi numerik
    for col in numerical_pca_1 + numerical_pca_2:
        input_df[[col]] = scalers[col].transform(input_df[[col]])

    # PCA
    pc1 = pca_1.transform(input_df[numerical_pca_1])
    pc2 = pca_2.transform(input_df[numerical_pca_2])
    pc1_df = pd.DataFrame(pc1, columns=[f"pc1_{i+1}" for i in range(pc1.shape[1])])
    pc2_df = pd.DataFrame(pc2, columns=[f"pc2_{i+1}" for i in range(pc2.shape[1])])

    # Gabungkan semua fitur akhir
    final_df = input_df[categorical_columns].copy()
    final_df = pd.concat([final_df, pc1_df, pc2_df], axis=1)

    try:
        pred = model.predict(final_df)
        pred_label = target_encoder.inverse_transform(pred)[0]
        st.success(f"🎯 Status Mahasiswa Diprediksi: {pred_label}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

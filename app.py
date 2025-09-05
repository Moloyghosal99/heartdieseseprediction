import streamlit as st
import pandas as pd
import pickle
import base64
import plotly.express as px

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="💠 Heart Disease Predictor",
    page_icon="💠",
    layout="wide"
)

# ----------------- CUSTOM CSS + HTML -----------------
st.markdown("""
    <style>
    /* Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e6f7ff 0%, #c3f2f5 50%, #f9f9f9 100%);
        color: black !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e0f7fa, #ffffff);
    }

    /* Card style */
    .stCard {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: black; /* Text always black */
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
    }

    h1, h2, h3, h4, h5 {
        color: #0072ff !important;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)
# ----------------- UTILITY FUNCTION -----------------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(['🔮 Predict', '📂 Bulk Predict', '📊 Model Information'])

# ---------------- TAB 1 : Single Prediction ----------------
with tab1:
    st.subheader("Give me your data")

    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type",
                              ["Typical Angina (TA)", "Atypical Angina (ATA)",
                               "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No (0)", "Yes (1)"])
    resting_ecg = st.selectbox("Resting ECG Results",
                               ["Normal", "ST-T Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No (N)", "Yes (Y)"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Mapping
    sex = 0 if sex == "Male" else 1
    chest_pain = {"Typical Angina (TA)": 0, "Atypical Angina (ATA)": 1,
                  "Non-Anginal Pain (NAP)": 2, "Asymptomatic (ASY)": 3}[chest_pain]
    fasting_bs = 1 if fasting_bs == "Yes (1)" else 0
    resting_ecg = {"Normal": 0, "ST-T Abnormality (ST)": 1, "Left Ventricular Hypertrophy (LVH)": 2}[resting_ecg]
    exercise_angina = 1 if exercise_angina == "Yes (Y)" else 0
    st_slope = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]

    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
        'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
    })

    algonames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl']

    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            predictions.append(model.predict(data))
        return predictions

    if st.button("Submit"):
        st.subheader('Results....')
        results = predict_heart_disease(input_data)

        for i, res in enumerate(results):
            st.subheader(algonames[i])
            if res[0] == 0:
                st.success("✅ No heart disease detected.")
            else:
                st.error("⚠️ Heart disease detected.")
            st.markdown('-----------------')

# ---------------- TAB 2 : Bulk Prediction ----------------
with tab2:
    st.title("📂 Upload CSV File")
    st.subheader("Instructions before uploading:")
    st.info("""
        1. No NaN values allowed.\n
        2. Exactly 11 features required in this order:\n
           'Age','Sex','ChestPainType','RestingBP','Cholesterol',
           'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'\n
        3. Ensure spelling and capitalization are correct.
    """)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file, encoding="cp1252")
        except Exception:
            uploaded_file.seek(0)
            input_data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        expected_columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol',
                            'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

        if list(input_data.columns) == expected_columns:
            model = pickle.load(open('LogisticR.pkl', 'rb'))
            input_data["Prediction LR"] = model.predict(input_data.values)
            st.subheader("Predictions:")
            st.dataframe(input_data)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please make sure the uploaded CSV has the correct columns (exact names).")
    else:
        st.info("📥 Upload a CSV file to get predictions.")

# ---------------- TAB 3 : Model Info ----------------
with tab3:
    st.subheader("📊 Model Performance")
    data = {'Decision Trees': 80.97,'Logistic Regression': 85.86,
            'Random Forest': 84.23,'Support Vector Machine': 84.22}
    df = pd.DataFrame(list(data.items()), columns=['Models', 'Accuracies'])

    fig = px.bar(df, x='Models', y='Accuracies',
                 title="Model Accuracies",
                 text='Accuracies',
                 color='Models',
                 animation_frame=None)

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside',
                      hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>")
    fig.update_layout(yaxis=dict(title="Accuracy (%)", range=[0, 100]),
                      transition={'duration': 500})

    st.plotly_chart(fig, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown('<div class="footer">⚡ Built with Streamlit | Made with 💠 for healthcare</div>', unsafe_allow_html=True)



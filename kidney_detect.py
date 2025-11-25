import pandas as pd
import numpy as np
import streamlit as st

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Kidney Disease Prediction App", layout="wide")

st.title("🩺 Kidney Disease Prediction App (CKD Detection)")
st.write("Machine Learning आधारित Kidney Disease classifier")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")

    # Keep only important columns
    important_columns = [
        'age', 'bp', 'sg', 'al', 'hemo', 'sc',
        'htn', 'dm', 'cad', 'appet', 'pc', 'classification'
    ]
    df = df[important_columns]

    # Clean object columns
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.replace("\t", "", regex=True)

    # Fill missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['bp'].fillna(df['bp'].median(), inplace=True)
    df['sg'].fillna(df['sg'].mode()[0], inplace=True)
    df['al'].fillna(df['al'].mode()[0], inplace=True)
    df['hemo'].fillna(df['hemo'].median(), inplace=True)
    df['sc'].fillna(df['sc'].median(), inplace=True)
    df['htn'].fillna(df['htn'].mode()[0], inplace=True)
    df['dm'].fillna(df['dm'].mode()[0], inplace=True)
    df['cad'].fillna(df['cad'].mode()[0], inplace=True)
    df['appet'].fillna(df['appet'].mode()[0], inplace=True)
    df['pc'].fillna(df['pc'].mode()[0], inplace=True)

    # Encoding
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

    return df

df = load_data()

# ----- ML Model -----
X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train[numeric_cols := ['age', 'bp', 'sg', 'al', 'hemo', 'sc']] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

model = AdaBoostClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("📊 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))

# ------ Sidebar User Input ------
st.sidebar.header("Enter Patient Details")

def get_user_input():
    age = st.sidebar.slider("Age", 1, 90, 45)
    bp = st.sidebar.slider("Blood Pressure", 50, 180, 80)
    sg = st.sidebar.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.sidebar.slider("Albumin", 0, 5, 1)
    hemo = st.sidebar.slider("Hemoglobin", 3.0, 17.0, 12.0)
    sc = st.sidebar.slider("Serum Creatinine", 0.1, 15.0, 1.2)

    htn = st.sidebar.selectbox("Hypertension", ['yes', 'no'])
    dm = st.sidebar.selectbox("Diabetes Mellitus", ['yes', 'no'])
    cad = st.sidebar.selectbox("Coronary Artery Disease", ['yes', 'no'])
    appet = st.sidebar.selectbox("Appetite", ['good', 'poor'])
    pc = st.sidebar.selectbox("Pus Cell", ['normal', 'abnormal'])

    data = {
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "hemo": hemo,
        "sc": sc,
        "htn": 1 if htn == 'yes' else 0,
        "dm": 1 if dm == 'yes' else 0,
        "cad": 1 if cad == 'yes' else 0,
        "appet": 1 if appet == 'good' else 0,
        "pc": 1 if pc == 'normal' else 0
    }

    return pd.DataFrame([data])

user_df = get_user_input()

# Scale numeric inputs
user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

# ----- Prediction -----
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(user_df)[0]
    
    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error("⚠️ Patient is likely to have **Chronic Kidney Disease (CKD)**.")
    else:
        st.success("✅ Patient does **NOT** have CKD.")

st.write("---")
st.write("Developed by **Jay Lamichhane**")

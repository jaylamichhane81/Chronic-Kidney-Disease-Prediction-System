import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")
    
    # Keep important columns only
    important_columns = ['age', 'bp', 'sg', 'al', 'hemo', 'sc', 
                         'htn', 'dm', 'cad', 'appet', 'pc', 'classification']
    df = df[important_columns]

    # Clean string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.replace('\t','', regex=True)

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

    # Encode categorical columns
    df['htn'] = df['htn'].map({'yes':1,'no':0})
    df['dm'] = df['dm'].map({'yes':1,'no':0})
    df['cad'] = df['cad'].map({'yes':1,'no':0})
    df['appet'] = df['appet'].map({'good':1,'poor':0})
    df['pc'] = df['pc'].map({'normal':1,'abnormal':0})
    df['classification'] = df['classification'].map({'ckd':1,'notckd':0})

    return df

df = load_data()

# ------------------------------
# Sidebar inputs
# ------------------------------
st.title("Chronic Kidney Disease Prediction")

st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", 1, 120, 25)
bp = st.sidebar.number_input("Blood Pressure", 50, 200, 80)
sg = st.sidebar.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.sidebar.selectbox("Albumin", [0,1,2,3,4,5])
hemo = st.sidebar.number_input("Hemoglobin", 5.0, 20.0, 15.0)
sc = st.sidebar.number_input("Serum Creatinine", 0.1, 10.0, 1.0)
htn = st.sidebar.selectbox("Hypertension", ["yes","no"])
dm = st.sidebar.selectbox("Diabetes Mellitus", ["yes","no"])
cad = st.sidebar.selectbox("Coronary Artery Disease", ["yes","no"])
appet = st.sidebar.selectbox("Appetite", ["good","poor"])
pc = st.sidebar.selectbox("Pus Cell", ["normal","abnormal"])

# ------------------------------
# Prepare input for model
# ------------------------------
input_data = pd.DataFrame({
    "age":[age],
    "bp":[bp],
    "sg":[sg],
    "al":[al],
    "hemo":[hemo],
    "sc":[sc],
    "htn":[1 if htn=="yes" else 0],
    "dm":[1 if dm=="yes" else 0],
    "cad":[1 if cad=="yes" else 0],
    "appet":[1 if appet=="good" else 0],
    "pc":[1 if pc=="normal" else 0]
})

# ------------------------------
# Scale numeric features
# ------------------------------
numeric_cols = ['age','bp','sg','al','hemo','sc']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# ------------------------------
# Train Model
# ------------------------------
X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = AdaBoostClassifier()
model.fit(X_train, y_train)

# ------------------------------
# Make Prediction
# ------------------------------
if st.button("Predict CKD"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("Warning: Patient may have Chronic Kidney Disease (CKD).")
    else:
        st.success("Patient is likely NOT to have CKD.")

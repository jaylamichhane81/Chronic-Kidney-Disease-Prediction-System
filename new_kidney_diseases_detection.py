# kidney_detect.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------
# Load dataset with caching
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('/content/kidney_disease.csv')
    
    # Keep important columns
    important_columns = ['age', 'bp', 'sg', 'al', 'hemo', 'sc','htn','dm','cad','appet','pc','classification']
    df = df[important_columns]

    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.replace('\t','', regex=True)
    
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

# ------------------------
# Streamlit UI
# ------------------------
st.title("Chronic Kidney Disease Prediction System")
st.write("Fill patient information to predict CKD risk")

# User input
age = st.number_input("Age", min_value=0, max_value=120, value=25)
bp = st.number_input("Blood Pressure (bp)", min_value=50, max_value=200, value=80)
sg = st.number_input("Specific Gravity (sg)", min_value=1.0, max_value=2.0, value=1.02, step=0.01)
al = st.number_input("Albumin (al)", min_value=0, max_value=5, value=0)
hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, max_value=20.0, value=15.0)
sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=20.0, value=1.0)

htn = st.selectbox("Hypertension (htn)", [0,1])
dm = st.selectbox("Diabetes Mellitus (dm)", [0,1])
cad = st.selectbox("Coronary Artery Disease (cad)", [0,1])
appet = st.selectbox("Appetite (appet)", [0,1])
pc = st.selectbox("Pus Cell (pc)", [0,1])

# ------------------------
# Prepare data for model
# ------------------------
X = df.drop("classification", axis=1)
y = df["classification"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
numeric_cols = ['age','bp','sg','al','hemo','sc']
scaler = MinMaxScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train model
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# ------------------------
# Prediction
# ------------------------
if st.button("Predict CKD"):
    input_data = pd.DataFrame({
        'age':[age],
        'bp':[bp],
        'sg':[sg],
        'al':[al],
        'hemo':[hemo],
        'sc':[sc],
        'htn':[htn],
        'dm':[dm],
        'cad':[cad],
        'appet':[appet],
        'pc':[pc]
    })

    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Patient is at risk of CKD")
    else:
        st.success("✅ Patient is NOT at risk of CKD")

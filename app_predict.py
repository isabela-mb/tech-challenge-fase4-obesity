import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =========================
# Config
# =========================
st.set_page_config(page_title="Preditor de Obesidade", layout="centered")
st.title("Sistema Preditivo de Obesidade")

DATA_PATH = "Obesity.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo `{DATA_PATH}` não encontrado no repositório.")
    st.stop()


# =========================
# Pré-processamento
# =========================
class RoundColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        for c in self.cols:
            if c in Xc.columns:
                Xc[c] = np.round(Xc[c]).astype(float)
        return Xc


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame) -> Pipeline:
    # Separar X/y
    if "Obesity" not in df.columns:
        raise ValueError("Coluna 'Obesity' não encontrada no dataset.")

    X = df.drop(columns=["Obesity"])
    y = df["Obesity"]

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    preprocess = Pipeline(steps=[
        ("rounder", RoundColumns(["FCVC", "NCP", "CH2O", "FAF", "TUE"])),
        ("ct", ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop"
        )),
    ])

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    pipe.fit(X, y)
    return pipe


df = load_dataset(DATA_PATH)

with st.spinner("Preparando modelo (primeira vez pode demorar um pouco)..."):
    model = train_model(df)

st.caption("Triagem preditiva com base em hábitos e medidas. Não substitui avaliação clínica.")

# =========================
# Formulário
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.number_input("Idade", min_value=14, max_value=61, value=25)

with col2:
    height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.75, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=80.0, step=0.1)

with col3:
    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])

st.divider()

favc = st.selectbox("Consumo frequente de alimentos calóricos (FAVC)", ["yes", "no"])
fcvc = st.slider("Consumo de vegetais (FCVC)", 1, 3, 2)
ncp = st.slider("Número de refeições principais (NCP)", 1, 4, 3)
caec = st.selectbox("Come entre refeições (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
ch2o = st.slider("Consumo de água (CH2O)", 1, 3, 2)
scc = st.selectbox("Monitora calorias (SCC)", ["yes", "no"])
faf = st.slider("Atividade física (FAF)", 0, 3, 2)
tue = st.slider("Tempo em dispositivos (TUE)", 0, 2, 1)
calc = st.selectbox("Consumo de álcool (CALC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Meio de transporte (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

if st.button("Prever", type="primary"):
    X_input = pd.DataFrame([{
        "Gender": gender,
        "Age": float(age),
        "Height": float(height),
        "Weight": float(weight),
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": float(ch2o),
        "SCC": scc,
        "FAF": float(faf),
        "TUE": float(tue),
        "CALC": calc,
        "MTRANS": mtrans
    }])

    pred = model.predict(X_input)[0]
    st.success(f"Predição: **{pred}**")

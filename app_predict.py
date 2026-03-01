import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Preditor de Obesidade", layout="centered")
st.title("Sistema Preditivo de Obesidade")
st.caption("Apoio à triagem clínica. Não substitui avaliação médica.")

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gênero", ["Female","Male"])
    age = st.number_input("Idade", min_value=14, max_value=61, value=25)
with col2:
    height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.75, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=80.0, step=0.1)
with col3:
    family_history = st.selectbox("Histórico familiar de excesso de peso", ["yes","no"])
    smoke = st.selectbox("Fuma?", ["yes","no"])

st.divider()
st.subheader("Hábitos alimentares e rotina")

favc = st.selectbox("Consome alimentos muito calóricos com frequência?", ["yes","no"])
fcvc = st.slider("Frequência de vegetais (1-3)", 1, 3, 2)
ncp = st.slider("Refeições principais por dia (1-4)", 1, 4, 3)
caec = st.selectbox("Come entre as refeições?", ["no","Sometimes","Frequently","Always"])
ch2o = st.slider("Consumo de água (1-3)", 1, 3, 2)
scc = st.selectbox("Monitora calorias?", ["yes","no"])
faf = st.slider("Atividade física semanal (0-3)", 0, 3, 2)
tue = st.slider("Tempo em dispositivos (0-2)", 0, 2, 1)
calc = st.selectbox("Consumo de álcool", ["no","Sometimes","Frequently","Always"])
mtrans = st.selectbox("Transporte habitual", ["Automobile","Motorbike","Bike","Public_Transportation","Walking"])

if st.button("Prever nível de obesidade", type="primary"):
    row = {
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
    }
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    st.success(f"Predição: **{pred}**")

st.divider()
st.markdown("**Observação:** protótipo educacional. Para produção: validação externa, auditoria de vieses e monitoramento.")

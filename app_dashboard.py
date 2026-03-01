import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard - Obesidade", layout="wide")
st.title("Dashboard Analítico — Obesidade (Tech Challenge Fase 04)")

df = pd.read_csv("Obesity.csv")

# Arredondar discretas com ruído
for col in ["FCVC","NCP","CH2O","FAF","TUE"]:
    df[col] = df[col].round()

st.caption("Objetivo: insights para a equipe médica.")

left, right = st.columns([2,1])

with left:
    st.subheader("Distribuição da variável alvo")
    fig = plt.figure()
    df["Obesity"].value_counts().plot(kind="bar")
    plt.xlabel("Classe")
    plt.ylabel("Qtde")
    st.pyplot(fig)

with right:
    st.subheader("Resumo rápido")
    st.write("Registros:", len(df))
    st.write("Idade média:", round(df["Age"].mean(), 2))
    st.write("Peso médio:", round(df["Weight"].mean(), 2))
    st.write("Altura média:", round(df["Height"].mean(), 2))

st.divider()

st.subheader("IMC (feature derivada) e obesidade")
df["BMI"] = df["Weight"] / (df["Height"]**2)

c1, c2 = st.columns(2)
with c1:
    fig = plt.figure()
    df.groupby("Obesity")["BMI"].mean().sort_values().plot(kind="barh")
    plt.xlabel("BMI médio")
    st.pyplot(fig)

with c2:
    fig = plt.figure()
    df.boxplot(column="BMI", by="Obesity", grid=False, rot=45)
    plt.title("")
    plt.suptitle("")
    plt.xlabel("Classe")
    plt.ylabel("BMI")
    st.pyplot(fig)

st.divider()

st.subheader("Hábitos vs Obesidade")
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("**Histórico familiar**")
    st.dataframe(pd.crosstab(df["family_history"], df["Obesity"], normalize="index"))

with colB:
    st.markdown("**Atividade física (FAF)**")
    st.dataframe(pd.crosstab(df["FAF"], df["Obesity"], normalize="index"))

with colC:
    st.markdown("**Alimentos calóricos (FAVC)**")
    st.dataframe(pd.crosstab(df["FAVC"], df["Obesity"], normalize="index"))

st.divider()
st.subheader("Insights prontos (visão de negócio)")
st.markdown("""
1. **IMC separa bem as classes**: BMI médio cresce consistentemente com o nível de obesidade.
2. **Histórico familiar** aparece associado a maior risco de sobrepeso/obesidade.
3. **Atividade física (FAF)** mais baixa tende a concentrar mais classes de obesidade.
4. **FAVC** (alimentos calóricos frequentes) se associa a sobrepeso/obesidade.
""")

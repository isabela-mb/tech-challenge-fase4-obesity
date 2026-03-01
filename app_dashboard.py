import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Dashboard — Obesidade", layout="wide")

DATA_PATH = "Obesity.csv"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1 {margin-bottom: 0.2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("Dashboard Analítico — Obesidade")

# -----------------------------
# Carregamento
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo `{DATA_PATH}` não encontrado.")
    st.stop()

@st.cache_data
def load_data(path: str):
    df_ = pd.read_csv(path)
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        if col in df_.columns:
            df_[col] = df_[col].round()
    df_["BMI"] = df_["Weight"] / (df_["Height"] ** 2)
    return df_

df = load_data(DATA_PATH)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filtros")

sex = st.sidebar.multiselect("Gênero", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
fh = st.sidebar.multiselect("Histórico familiar", sorted(df["family_history"].unique()), default=sorted(df["family_history"].unique()))
favc = st.sidebar.multiselect("FAVC", sorted(df["FAVC"].unique()), default=sorted(df["FAVC"].unique()))
mtrans = st.sidebar.multiselect("Transporte", sorted(df["MTRANS"].unique()), default=sorted(df["MTRANS"].unique()))

age_range = st.sidebar.slider("Idade",
                              int(df["Age"].min()),
                              int(df["Age"].max()),
                              (int(df["Age"].min()), int(df["Age"].max())))

bmi_range = st.sidebar.slider("BMI",
                              float(np.floor(df["BMI"].min())),
                              float(np.ceil(df["BMI"].max())),
                              (float(np.floor(df["BMI"].min())),
                               float(np.ceil(df["BMI"].max()))))

df_f = df[
    (df["Gender"].isin(sex)) &
    (df["family_history"].isin(fh)) &
    (df["FAVC"].isin(favc)) &
    (df["MTRANS"].isin(mtrans)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["BMI"].between(bmi_range[0], bmi_range[1]))
].copy()

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Registros", f"{len(df_f):,}".replace(",", "."))
c2.metric("Classes", df_f["Obesity"].nunique())
c3.metric("Idade média", f"{df_f['Age'].mean():.1f}")
c4.metric("BMI médio", f"{df_f['BMI'].mean():.1f}")
c5.metric("Peso médio", f"{df_f['Weight'].mean():.1f}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "IMC", "Hábitos", "Correlação"])

with tab1:
    vc = df_f["Obesity"].value_counts().reset_index()
    vc.columns = ["Obesity", "count"]

    fig = px.bar(vc, x="Obesity", y="count")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df_f, x="Age", y="BMI", color="Obesity", opacity=0.7)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df_f, x="Obesity", y="BMI")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df_f, x="BMI", color="Obesity", nbins=40, barmode="overlay")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    colA, colB = st.columns(2)

    with colA:
        g = df_f.groupby(["family_history", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g["count"] / g.groupby("family_history")["count"].transform("sum")

        fig = px.bar(g, x="family_history", y="pct", color="Obesity")
        fig.update_layout(yaxis_tickformat=".0%", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        g = df_f.groupby(["FAF", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g["count"] / g.groupby("FAF")["count"].transform("sum")

        fig = px.bar(g, x="FAF", y="pct", color="Obesity")
        fig.update_layout(yaxis_tickformat=".0%", height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
    num_cols = [c for c in num_cols if c in df_f.columns]

    corr = df_f[num_cols].corr(numeric_only=True)

    fig = px.imshow(corr, text_auto=".2f")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

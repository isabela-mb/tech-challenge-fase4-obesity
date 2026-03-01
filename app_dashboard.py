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
/* Deixa o topo mais “clean” e melhora espaçamento */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* Título com mais presença */
h1 {margin-bottom: 0.2rem;}
/* Cards (métrica) mais alinhados */
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("Dashboard Analítico — Obesidade")
st.caption("Tech Challenge Fase 04 • Exploração e insights para apoio à decisão")

# -----------------------------
# Carregamento com validação
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo `{DATA_PATH}` não encontrado na raiz do repositório.")
    st.info("Suba o `Obesity.csv` para o GitHub no mesmo nível do `app_dashboard.py`.")
    st.stop()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    # arredondar discretas com ruído
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        if col in df_.columns:
            df_[col] = df_[col].round()
    # feature derivada
    df_["BMI"] = df_["Weight"] / (df_["Height"] ** 2)
    return df_

df = load_data(DATA_PATH)

# -----------------------------
# Sidebar (filtros)
# -----------------------------
st.sidebar.header("Filtros")

sex = st.sidebar.multiselect("Gênero", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
fh = st.sidebar.multiselect("Histórico familiar", sorted(df["family_history"].unique()), default=sorted(df["family_history"].unique()))
favc = st.sidebar.multiselect("Alimentos calóricos (FAVC)", sorted(df["FAVC"].unique()), default=sorted(df["FAVC"].unique()))
mtrans = st.sidebar.multiselect("Transporte (MTRANS)", sorted(df["MTRANS"].unique()), default=sorted(df["MTRANS"].unique()))

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Idade", min_value=age_min, max_value=age_max, value=(age_min, age_max))

bmi_min, bmi_max = float(df["BMI"].min()), float(df["BMI"].max())
bmi_range = st.sidebar.slider("BMI", min_value=float(np.floor(bmi_min)), max_value=float(np.ceil(bmi_max)),
                              value=(float(np.floor(bmi_min)), float(np.ceil(bmi_max))))

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
tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "IMC & Distribuições", "Hábitos", "Correlações"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        vc = df_f["Obesity"].value_counts().reset_index()
        vc.columns = ["Obesity", "count"]
        fig = px.bar(vc, x="Obesity", y="count", title="Distribuição do Nível de Obesidade (alvo)")
        fig.update_layout(xaxis_title="", yaxis_title="Quantidade", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Insights rápidos")
        st.markdown("""
- **O BMI (IMC) é o maior separador** entre as classes (tendência monotônica).
- **Histórico familiar** aparece como fator de risco importante.
- **Hábitos (FAF, FAVC, CAEC)** ajudam a explicar sobrepeso vs obesidade.
        """)
        st.caption("Dica de apresentação: use esses bullets direto no slide.")

    st.subheader("Mapa de risco (BMI x Idade)")
    fig = px.scatter(
        df_f, x="Age", y="BMI", color="Obesity",
        hover_data=["Gender", "family_history", "FAVC", "FAF", "MTRANS"],
        opacity=0.7, title="Dispersão: Idade vs BMI (colorido por classe)"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df_f, x="Obesity", y="BMI", title="Boxplot: BMI por classe")
        fig.update_layout(xaxis_title="", yaxis_title="BMI", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df_f, x="BMI", color="Obesity", nbins=40, title="Distribuição de BMI por classe", barmode="overlay")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df_f, x="Age", nbins=30, title="Distribuição de Idade")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(df_f, x="Weight", nbins=30, title="Distribuição de Peso (kg)")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Proporção das classes por hábito (normalizado por grupo)")

    colA, colB = st.columns(2)
    with colA:
        g = df_f.groupby(["family_history", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g.groupby("family_history")["count"].apply(lambda s: s / s.sum())
        fig = px.bar(g, x="family_history", y="pct", color="Obesity", title="Histórico familiar vs classes", text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Proporção", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        g = df_f.groupby(["FAVC", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g.groupby("FAVC")["count"].apply(lambda s: s / s.sum())
        fig = px.bar(g, x="FAVC", y="pct", color="Obesity", title="FAVC (calóricos) vs classes", text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Proporção", height=420)
        st.plotly_chart(fig, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        g = df_f.groupby(["FAF", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g.groupby("FAF")["count"].apply(lambda s: s / s.sum())
        fig = px.bar(g, x="FAF", y="pct", color="Obesity", title="FAF (atividade física) vs classes", text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="FAF (0–3)", yaxis_title="Proporção", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with colD:
        g = df_f.groupby(["CAEC", "Obesity"]).size().reset_index(name="count")
        g["pct"] = g.groupby("CAEC")["count"].apply(lambda s: s / s.sum())
        fig = px.bar(g, x="CAEC", y="pct", color="Obesity", title="CAEC (lanches entre refeições) vs classes", text_auto=".0%")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Proporção", height=420)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Correlação (somente variáveis numéricas)")
    num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
    num_cols = [c for c in num_cols if c in df_f.columns]
    corr = df_f[num_cols].corr(numeric_only=True)

    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlação")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perguntas que esse painel responde (pra sua apresentação)")
    st.markdown("""
- Quais classes de obesidade são mais frequentes no recorte filtrado?
- Como **BMI** varia por classe (distribuição e dispersão)?
- Quais hábitos parecem associados a maior proporção de obesidade (FAVC, FAF, CAEC, histórico familiar)?
- Quais variáveis numéricas têm maior correlação entre si (e com BMI)?
    """)

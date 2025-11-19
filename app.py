
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="AgroIA Dashboard", page_icon="🌾", layout="wide")

# Título
st.title("🚜 AgroIA: Sistema de Recomendación de Cultivos")
st.markdown("Modelo de IA entrenado para clasificar cultivos según N, P, K y pH.")

# Carga de datos
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("soil_measures.csv")
        X = df.drop('crop', axis=1)
        y = df['crop']
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        model.fit(X, y)
        return model
    except FileNotFoundError:
        return None

model = load_data()

if model:
    # Sidebar
    st.sidebar.header("Parámetros del Suelo")
    N = st.sidebar.slider('Nitrógeno (N)', 0, 140, 50)
    P = st.sidebar.slider('Fósforo (P)', 5, 145, 50)
    K = st.sidebar.slider('Potasio (K)', 5, 205, 50)
    ph = st.sidebar.slider('pH', 0.0, 14.0, 6.5)
    
    # Predicción
    if st.button("Analizar Suelo"):
        input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
        prediction = model.predict(input_data)[0]
        proba = np.max(model.predict_proba(input_data))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Cultivo Recomendado: **{prediction.upper()}**")
        with col2:
            st.metric("Confianza", f"{proba:.1%}")
else:
    st.error("Error: No se encontró el archivo 'soil_measures.csv'.")

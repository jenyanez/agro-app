
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="AgroDecision AI", page_icon="📊", layout="wide")
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- MODELO ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("soil_measures.csv")
        X = df.drop('crop', axis=1)
        y = df['crop']
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        model.fit(X, y)
        return model, df
    except:
        return None, None

model, df_source = load_data()

# --- UI ---
st.sidebar.title("🎛️ Panel de Control")
st.sidebar.info("Ajuste los parámetros edafológicos abajo:")

N = st.sidebar.number_input("Nitrógeno (N)", 0, 140, 50)
P = st.sidebar.number_input("Fósforo (P)", 5, 145, 50)
K = st.sidebar.number_input("Potasio (K)", 5, 205, 50)
ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5, 0.1)
rain = st.sidebar.number_input("Lluvia (mm)", 0, 300, 100)

if st.sidebar.button("Analizar Viabilidad", type="primary"):
    if model:
        input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
        prediction = model.predict(input_data)[0]
        proba = np.max(model.predict_proba(input_data))
        
        # Gap Analysis
        crop_stats = df_source[df_source['crop'] == prediction].mean(numeric_only=True)
        
        st.title(f"Resultados: {prediction.upper()}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Confianza IA", f"{proba:.1%}")
            status_water = "Óptimo" if rain > 100 else "Riesgo Sequía"
            st.metric("Hidrología", status_water)
            
            if (N - crop_stats['N']) < -10:
                st.warning(f"📉 Déficit de Nitrógeno detectado. Aplique Urea.")
            else:
                st.success("✅ Nutrientes Balanceados")

        with col2:
            st.subheader("🔬 Análisis de Radar")
            categories = ['Nitrógeno', 'Fósforo', 'Potasio', 'pH']
            fig = go.Figure()
            
            # Input del Usuario
            fig.add_trace(go.Scatterpolar(
                r=[N, P, K, ph*10], 
                theta=categories, 
                fill='toself', 
                name='Tu Suelo'
            ))
            
            # Ideal del Cultivo
            fig.add_trace(go.Scatterpolar(
                r=[crop_stats['N'], crop_stats['P'], crop_stats['K'], crop_stats['ph']*10], 
                theta=categories, 
                fill='toself', 
                name='Ideal'
            ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 250])), height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Error cargando modelo.")
else:
    st.info("👈 Inicie el análisis desde el menú lateral.")

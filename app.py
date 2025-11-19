
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

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
        # 1. PREDICCIÓN
        input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
        prediction = model.predict(input_data)[0]
        proba = np.max(model.predict_proba(input_data))
        
        # 2. GAP ANALYSIS (MEJORA #2)
        crop_stats = df_source[df_source['crop'] == prediction].mean(numeric_only=True)
        
        # Calculamos gaps
        gap_n = N - crop_stats['N']
        gap_p = P - crop_stats['P']
        gap_k = K - crop_stats['K']
        
        # UI RESULTADOS
        st.title(f"Resultados del Análisis: {prediction.upper()}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Confianza IA", f"{proba:.1%}")
            status_water = "Óptimo" if rain > 100 else "Riesgo Sequía"
            st.metric("Hidrología", status_water)
            
            st.markdown("#### 📋 Diagnóstico de Nutrientes")
            # Lógica de alertas
            if gap_n < -10:
                st.warning(f"📉 **Nitrógeno Bajo:** {N} (Ideal: {crop_stats['N']:.0f}). Aplique Urea.")
            elif gap_n > 20:
                st.info(f"📈 **Exceso Nitrógeno:** Reduzca fertilización.")
            else:
                st.success("✅ Nitrógeno Balanceado")
                
            if gap_p < -10:
                st.warning(f"📉 **Fósforo Bajo:** {P} (Ideal: {crop_stats['P']:.0f}). Aplique Fosfato.")
            
            # --- GENERACIÓN DE REPORTE (MEJORA #3) ---
            st.markdown("---")
            st.markdown("#### 📄 Exportar Datos")
            
            # Creamos un DataFrame pequeño para el reporte
            report_data = pd.DataFrame({
                'Parametro': ['Cultivo Predicho', 'Confianza', 'Nitrogeno (Input)', 'Fosforo (Input)', 'Potasio (Input)', 'pH (Input)', 'Lluvia (Input)', 'Estado Hidrico'],
                'Valor': [prediction.upper(), f"{proba:.1%}", N, P, K, ph, rain, status_water]
            })
            
            # Convertimos a CSV
            csv_report = report_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Descargar Informe Técnico (CSV)",
                data=csv_report,
                file_name=f"Reporte_AgroIA_{prediction}_{int(time.time())}.csv",
                mime='text/csv',
            )

        with col2:
            st.subheader("🔬 Análisis de Radar (MEJORA #1)")
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
                name='Ideal Promedio'
            ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 250])), height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("*pH escalado x10 para visualización")
            
    else:
        st.error("Error cargando modelo.")
else:
    st.info("👈 Inicie el análisis desde el menú lateral.")

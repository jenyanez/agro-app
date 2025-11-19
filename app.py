
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AgroDecision System", layout="wide")
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

# --- UI: SIDEBAR (Sin iconos infantiles) ---
st.sidebar.title("Panel de Control")
st.sidebar.markdown("**Parámetros Edafológicos**")

N = st.sidebar.number_input("Nitrógeno (N)", 0, 140, 50)
P = st.sidebar.number_input("Fósforo (P)", 5, 145, 50)
K = st.sidebar.number_input("Potasio (K)", 5, 205, 50)
ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5, 0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("**Variables Hidrológicas**")
rain = st.sidebar.number_input("Precipitación Media (mm)", 0, 300, 100)

if st.sidebar.button("Ejecutar Análisis", type="primary"):
    if model:
        # 1. PREDICCIÓN
        input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
        prediction = model.predict(input_data)[0]
        proba = np.max(model.predict_proba(input_data))
        
        # 2. GAP ANALYSIS
        crop_stats = df_source[df_source['crop'] == prediction].mean(numeric_only=True)
        
        # Calculamos desviaciones
        gap_n = N - crop_stats['N']
        gap_p = P - crop_stats['P']
        
        # UI RESULTADOS
        st.title(f"Reporte de Viabilidad: {prediction.upper()}")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Métricas Clave")
            st.metric("Índice de Confianza (Modelo)", f"{proba:.1%}")
            
            status_water = "Adecuado" if rain > 100 else "Déficit Hídrico"
            st.metric("Disponibilidad Hídrica", status_water)
            
            st.markdown("#### Diagnóstico de Nutrientes")
            # Alertas limpias sin emojis excesivos
            if gap_n < -10:
                st.warning(f"Déficit de Nitrógeno: {N} (Objetivo: {crop_stats['N']:.0f})")
            elif gap_n > 20:
                st.info(f"Exceso de Nitrógeno detectado")
            else:
                st.success("Niveles de Nitrógeno óptimos")
                
            # Exportar Reporte
            st.markdown("---")
            report_data = pd.DataFrame({
                'Variable': ['Cultivo', 'Confianza', 'Estado Hídrico'],
                'Valor': [prediction.upper(), f"{proba:.1%}", status_water]
            })
            csv = report_data.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Reporte CSV", csv, "reporte_agronomia.csv", "text/csv")

        with col2:
            st.subheader("Análisis Comparativo de Nutrientes")
            
            categories = ['Nitrógeno', 'Fósforo', 'Potasio', 'pH (x10)']
            
            fig = go.Figure()
            
            # Trazado del Ideal (Fondo de referencia)
            fig.add_trace(go.Scatterpolar(
                r=[crop_stats['N'], crop_stats['P'], crop_stats['K'], crop_stats['ph']*10], 
                theta=categories, 
                fill='toself', 
                name='Estándar Ideal',
                line_color='rgba(128, 128, 128, 0.5)', # Gris profesional
                fillcolor='rgba(128, 128, 128, 0.2)',
                hoverinfo='skip'
            ))
            
            # Trazado del Usuario (Dato principal)
            fig.add_trace(go.Scatterpolar(
                r=[N, P, K, ph*10], 
                theta=categories, 
                fill='toself', 
                name='Muestra Actual',
                line_color='#003366', # Azul Navy Corporativo
                fillcolor='rgba(0, 51, 102, 0.4)',
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 250], gridcolor='lightgrey'),
                    bgcolor='white'
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=40, r=40, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.error("Error: Modelo no cargado.")
else:
    st.info("Configure los parámetros en el panel lateral para iniciar el análisis.")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="AgroDecision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para limpiar la interfaz (Ocultar elementos innecesarios)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# --- REGLAS DE NEGOCIO ---
CROP_WATER_NEEDS = {
    'high': ['rice', 'jute', 'coconut', 'papaya'],
    'moderate': ['coffee', 'banana', 'maize', 'cotton'], 
    'low': ['chickpea', 'kidneybeans', 'mothbeans', 'mungbean', 'blackgram', 'lentil']
}

# --- MODELO ---
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("soil_measures.csv")
        X = df.drop('crop', axis=1)
        y = df['crop']
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        model.fit(X, y)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- INTERFAZ CORPORATIVA ---
st.title("Sistema de Soporte a la Decisión Agrícola")
st.markdown("Análisis predictivo de viabilidad de cultivos basado en parámetros edafológicos y restricciones hídricas.")
st.markdown("---")

if model:
    # Layout de 3 columnas: 1 Sidebar (Inputs), 2 Panel Central (Resultados)
    col_inputs, col_results = st.columns([1, 2])

    with col_inputs:
        st.subheader("Parámetros de Entrada")
        
        with st.expander("Perfil Edafológico (Suelo)", expanded=True):
            N = st.number_input('Nitrógeno (N)', 0, 140, 50)
            P = st.number_input('Fósforo (P)', 5, 145, 50)
            K = st.number_input('Potasio (K)', 5, 205, 50)
            ph = st.slider('pH', 0.0, 14.0, 6.5, step=0.1)
            
        with st.expander("Restricciones Hidrológicas", expanded=True):
            rainfall = st.number_input('Precipitación Media (mm)', 0, 300, 100)

        run_btn = st.button("Calcular Viabilidad", type="primary", use_container_width=True)

    with col_results:
        if run_btn:
            # Lógica
            input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
            prediction = model.predict(input_data)[0]
            proba = np.max(model.predict_proba(input_data))
            
            # Validación Hídrica
            status = "OPTIMAL"
            msg = "Condiciones ideales."
            
            if prediction in CROP_WATER_NEEDS['high'] and rainfall < 150:
                status = "WARNING"
                msg = f"Déficit hídrico detectado ({rainfall}mm). Se requiere riego suplementario."
            elif prediction in CROP_WATER_NEEDS['low'] and rainfall > 200:
                status = "WARNING"
                msg = f"Exceso hídrico detectado ({rainfall}mm). Riesgo de saturación radicular."

            # --- DASHBOARD DE RESULTADOS ---
            st.subheader("Reporte de Análisis")
            
            # Fila de Métricas
            m1, m2, m3 = st.columns(3)
            m1.metric("Cultivo Identificado", prediction.upper())
            m2.metric("Índice de Confianza (IA)", f"{proba:.1%}")
            m3.metric("Estado Hídrico", status)

            # Alertas visuales limpias
            if status == "OPTIMAL":
                st.success(f"✅ Viabilidad Confirmada: El perfil es apto para **{prediction.upper()}**.")
            else:
                st.warning(f"⚠️ Alerta Operativa: {msg}")

            # Gráfico de distribución
            st.markdown("#### Distribución de Probabilidades")
            probs = model.predict_proba(input_data)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            chart_data = pd.DataFrame({
                'Cultivo': model.classes_[top3_idx], 
                'Score': probs[top3_idx]
            })
            st.bar_chart(chart_data.set_index('Cultivo'))

        else:
            st.info("Ingrese los parámetros en el panel izquierdo y presione 'Calcular'.")

else:
    st.error("Error de sistema: Base de datos de suelos no disponible.")

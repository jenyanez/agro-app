
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="AgroIA Pro", page_icon="🚜", layout="wide")

# --- DICCIONARIO DE REGLAS AGRONÓMICAS (SISTEMA EXPERTO) ---
# Clasificación simple de requerimientos hídricos para validación cruzada
CROP_WATER_NEEDS = {
    'high': ['rice', 'jute', 'coconut', 'papaya'],  # Necesitan mucha agua (>150mm)
    'moderate': ['coffee', 'banana', 'maize', 'cotton'], 
    'low': ['chickpea', 'kidneybeans', 'mothbeans', 'mungbean', 'blackgram', 'lentil'] # Secano
}

# --- CARGA DE MODELO ---
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

# --- INTERFAZ DE USUARIO ---
st.title("🚜 AgroIA: Sistema Híbrido de Recomendación")
st.markdown("""
Este sistema combina **Machine Learning** (para análisis edafológico) con **Reglas Expertas** (para viabilidad climática) para sugerir el cultivo óptimo.
""")

if model:
    col_config, col_pred = st.columns([1, 2])

    with col_config:
        st.header("1. Parámetros")
        st.subheader("🧪 Química del Suelo (Input IA)")
        N = st.slider('Nitrógeno (N)', 0, 140, 50, help="Ratio de contenido de Nitrógeno en el suelo")
        P = st.slider('Fósforo (P)', 5, 145, 50)
        K = st.slider('Potasio (K)', 5, 205, 50)
        ph = st.slider('pH del Suelo', 0.0, 14.0, 6.5)
        
        st.subheader("🌧️ Condiciones Climáticas (Input Experto)")
        # Este dato NO entra al modelo, entra a la capa de lógica de negocio
        rainfall = st.slider('Disponibilidad de Agua/Lluvia (mm)', 0, 300, 100, 
                             help="Promedio de lluvia o capacidad de riego disponible.")

    with col_pred:
        st.header("2. Análisis de Viabilidad")
        
        if st.button("Ejecutar Análisis", type="primary"):
            # A. PREDICCIÓN DEL MODELO (Based on Soil)
            input_data = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})
            prediction = model.predict(input_data)[0]
            proba = np.max(model.predict_proba(input_data))
            
            # B. LÓGICA DE NEGOCIO (Validación Hídrica)
            water_status = "OK"
            warning_msg = ""
            
            # Verificamos si el cultivo predicho tiene requisitos especiales
            if prediction in CROP_WATER_NEEDS['high']:
                if rainfall < 150:
                    water_status = "RISK"
                    warning_msg = f"⚠️ **ALERTA AGRONÓMICA:** El suelo es ideal para **{prediction.upper()}**, pero la disponibilidad de agua ({rainfall}mm) es insuficiente. Requiere >150mm o riego artificial."
            
            elif prediction in CROP_WATER_NEEDS['low']:
                if rainfall > 200:
                    water_status = "RISK"
                    warning_msg = f"⚠️ **RIESGO DE PUDRICIÓN:** El suelo sugiere **{prediction.upper()}**, pero el exceso de agua ({rainfall}mm) podría dañar la raíz. Se recomienda drenaje."

            # C. MOSTRAR RESULTADOS
            st.divider()
            
            if water_status == "OK":
                st.success(f"✅ Cultivo Óptimo: **{prediction.upper()}**")
                st.caption(f"El perfil de suelo y agua son compatibles. (Confianza del modelo: {proba:.1%})")
            else:
                # Si hay conflicto, mostramos el cultivo pero con advertencia amarilla/naranja
                st.warning(f"⚠️ Cultivo Sugerido por Suelo: **{prediction.upper()}**")
                st.info(warning_msg)
                st.caption(f"Confianza química del modelo: {proba:.1%}")

            # Gráfico simple de probabilidad
            probs = model.predict_proba(input_data)
            top3_idx = np.argsort(probs[0])[-3:][::-1]
            chart_data = pd.DataFrame({
                'Cultivo': model.classes_[top3_idx], 
                'Probabilidad': probs[0][top3_idx]
            })
            st.bar_chart(chart_data.set_index('Cultivo'))

else:
    st.error("No se encontraron los datos. Por favor carga 'soil_measures.csv'.")

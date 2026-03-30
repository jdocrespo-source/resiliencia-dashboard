import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Sistema Predictivo de Resiliencia Empresarial",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "modelo_resiliencia.pkl"
DATA_PATH = "dataset_resiliencia_pymes.xlsx"

FEATURES = [
    "Liquidez",
    "ROA",
    "Endeudamiento",
    "Margen_Neto",
    "Crec_Ventas",
    "Adapt_Digital",
    "Ventaja_Competitiva"
]

# =========================================================
# FUNCIONES
# =========================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    return modelo

@st.cache_data
def cargar_dataset():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

def clasificar_riesgo(prob):
    if prob >= 0.80:
        return "Bajo", "🟢 Verde", "La empresa presenta una condición favorable frente a escenarios adversos."
    elif prob >= 0.60:
        return "Moderado", "🟡 Amarillo", "La empresa muestra resiliencia intermedia, pero requiere fortalecimiento."
    else:
        return "Alto", "🔴 Rojo", "La empresa presenta alta vulnerabilidad ante cambios del entorno."

def diagnostico_reglas(liquidez, roa, endeudamiento, margen_neto, crec_ventas, adapt_digital, ventaja_competitiva):
    fortalezas = []
    alertas = []

    if liquidez < 1.2:
        alertas.append("Liquidez insuficiente")
    else:
        fortalezas.append("Liquidez adecuada")

    if endeudamiento > 0.65:
        alertas.append("Endeudamiento elevado")
    else:
        fortalezas.append("Endeudamiento controlado")

    if roa < 0.03:
        alertas.append("ROA bajo")
    else:
        fortalezas.append("ROA favorable")

    if margen_neto < 0.05:
        alertas.append("Margen neto reducido")
    else:
        fortalezas.append("Margen neto saludable")

    if crec_ventas < 0:
        alertas.append("Crecimiento de ventas negativo")
    elif crec_ventas < 0.03:
        alertas.append("Crecimiento de ventas débil")
    else:
        fortalezas.append("Crecimiento comercial positivo")

    if adapt_digital < 5:
        alertas.append("Baja adaptación digital")
    else:
        fortalezas.append("Adaptación digital favorable")

    if ventaja_competitiva < 5:
        alertas.append("Ventaja competitiva limitada")
    else:
        fortalezas.append("Ventaja competitiva sólida")

    return fortalezas, alertas

def recomendacion_por_riesgo(riesgo):
    if riesgo == "Bajo":
        return "Mantener la estrategia actual y monitorear periódicamente los indicadores financieros y estratégicos."
    elif riesgo == "Moderado":
        return "Fortalecer liquidez, adaptación digital y ventaja competitiva para mejorar la resiliencia."
    else:
        return "Aplicar un plan correctivo inmediato sobre endeudamiento, liquidez y capacidades estratégicas."

def convertir_a_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicciones")
    output.seek(0)
    return output

# =========================================================
# CARGA DE RECURSOS
# =========================================================
try:
    model = cargar_modelo()
    df_base = cargar_dataset()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    error_inicio = str(e)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Panel de control")
st.sidebar.markdown("Sistema predictivo de resiliencia empresarial para PYMES")
st.sidebar.markdown("---")

if modelo_ok:
    st.sidebar.success("Modelo cargado correctamente")
    st.sidebar.write(f"Archivo modelo: `{MODEL_PATH}`")
    st.sidebar.write(f"Archivo datos: `{DATA_PATH}`")
else:
    st.sidebar.error("Error al cargar el modelo o el dataset")

# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
st.title("📊 Sistema Predictivo de Resiliencia Empresarial para PYMES")
st.markdown("Dashboard ejecutivo para apoyar la toma de decisiones mediante machine learning.")

if not modelo_ok:
    st.error("No fue posible iniciar el dashboard.")
    st.code(error_inicio)
    st.stop()

st.success("Modelo final cargado desde archivo `.pkl`")

tab1, tab2, tab3 = st.tabs(["Simulación individual", "Predicción masiva", "Base de referencia"])

# =========================================================
# TAB 1: SIMULACIÓN INDIVIDUAL
# =========================================================
with tab1:
    st.subheader("Simulación individual de empresa")

    col1, col2 = st.columns(2)

    with col1:
        liquidez = st.number_input("Liquidez", value=1.80, format="%.4f")
        roa = st.number_input("ROA", value=0.06, format="%.4f")
        endeudamiento = st.number_input("Endeudamiento", value=0.45, format="%.4f")
        margen_neto = st.number_input("Margen_Neto", value=0.08, format="%.4f")

    with col2:
        crec_ventas = st.number_input("Crec_Ventas", value=0.04, format="%.4f")
        adapt_digital = st.number_input("Adapt_Digital", value=7.0, format="%.4f")
        ventaja_competitiva = st.number_input("Ventaja_Competitiva", value=7.0, format="%.4f")

    if st.button("Analizar empresa"):
        entrada = pd.DataFrame([{
            "Liquidez": liquidez,
            "ROA": roa,
            "Endeudamiento": endeudamiento,
            "Margen_Neto": margen_neto,
            "Crec_Ventas": crec_ventas,
            "Adapt_Digital": adapt_digital,
            "Ventaja_Competitiva": ventaja_competitiva
        }])

        pred = model.predict(entrada)[0]
        prob = model.predict_proba(entrada)[0][1]

        clasificacion = "RESILIENTE" if pred == 1 else "NO RESILIENTE"
        riesgo, semaforo, mensaje = clasificar_riesgo(prob)
        fortalezas, alertas = diagnostico_reglas(
            liquidez, roa, endeudamiento, margen_neto,
            crec_ventas, adapt_digital, ventaja_competitiva
        )
        recomendacion = recomendacion_por_riesgo(riesgo)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Clasificación", clasificacion)
        r2.metric("Probabilidad", f"{prob:.2%}")
        r3.metric("Riesgo", riesgo)
        r4.metric("Semáforo", semaforo)

        st.markdown("### Diagnóstico ejecutivo")
        st.write(mensaje)

        st.markdown("### Fortalezas detectadas")
        if fortalezas:
            for f in fortalezas:
                st.write(f"- {f}")
        else:
            st.write("No se identifican fortalezas destacadas.")

        st.markdown("### Alertas detectadas")
        if alertas:
            for a in alertas:
                st.write(f"- {a}")
        else:
            st.write("No se detectan alertas críticas.")

        st.markdown("### Recomendación estratégica")
        if riesgo == "Bajo":
            st.success(recomendacion)
        elif riesgo == "Moderado":
            st.warning(recomendacion)
        else:
            st.error(recomendacion)

# =========================================================
# TAB 2: PREDICCIÓN MASIVA
# =========================================================
with tab2:
    st.subheader("Predicción masiva desde archivo Excel")

    st.markdown(
        """
        Sube un archivo Excel con las siguientes columnas:
        - Liquidez
        - ROA
        - Endeudamiento
        - Margen_Neto
        - Crec_Ventas
        - Adapt_Digital
        - Ventaja_Competitiva
        """
    )

    archivo_subido = st.file_uploader("Subir archivo Excel", type=["xlsx"])

    if archivo_subido is not None:
        try:
            df_nuevo = pd.read_excel(archivo_subido)
            df_nuevo.columns = df_nuevo.columns.str.strip()

            faltantes = [c for c in FEATURES if c not in df_nuevo.columns]
            if faltantes:
                st.error(f"Faltan columnas requeridas en el archivo cargado: {faltantes}")
            else:
                df_pred = df_nuevo.copy()

                for col in FEATURES:
                    df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

                df_pred_limpio = df_pred.dropna(subset=FEATURES).copy()

                if df_pred_limpio.empty:
                    st.error("El archivo cargado no contiene filas válidas después de la limpieza.")
                else:
                    probs = model.predict_proba(df_pred_limpio[FEATURES])[:, 1]
                    preds = model.predict(df_pred_limpio[FEATURES])

                    df_pred_limpio["Prediccion"] = np.where(preds == 1, "RESILIENTE", "NO RESILIENTE")
                    df_pred_limpio["Probabilidad_Resiliencia"] = probs

                    riesgos = []
                    semaforos = []
                    recomendaciones = []

                    for p in probs:
                        riesgo, semaforo, _ = clasificar_riesgo(p)
                        riesgos.append(riesgo)
                        semaforos.append(semaforo)
                        recomendaciones.append(recomendacion_por_riesgo(riesgo))

                    df_pred_limpio["Nivel_Riesgo"] = riesgos
                    df_pred_limpio["Semaforo"] = semaforos
                    df_pred_limpio["Recomendacion"] = recomendaciones

                    st.success("Predicciones generadas correctamente")
                    st.dataframe(df_pred_limpio, use_container_width=True)

                    archivo_excel = convertir_a_excel(df_pred_limpio)

                    st.download_button(
                        label="📥 Descargar resultados en Excel",
                        data=archivo_excel,
                        file_name="predicciones_resiliencia_pymes.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error("No fue posible procesar el archivo cargado.")
            st.code(str(e))

# =========================================================
# TAB 3: BASE DE REFERENCIA
# =========================================================
with tab3:
    st.subheader("Vista previa de la base de referencia")
    st.dataframe(df_base.head(30), use_container_width=True)
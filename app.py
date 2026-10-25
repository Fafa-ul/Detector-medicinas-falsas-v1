"""
App Streamlit: detector de lecturas atípicas de OCR en medicinas.
Usa el Random Forest entrenado en Orange (archivo .pkcls).

Ejecutar:  streamlit run app.py
"""
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from caracteristicas import NOMBRES_VARIABLES, calcular_variables

RUTA_MODELO = Path(__file__).parent / "modelo_ocr.pkcls"

st.set_page_config(page_title="Detector de lecturas atípicas", page_icon="💊")


# ---------------------------------------------------------------- modelo
@st.cache_resource
def cargar_modelo(datos: bytes):
    """Carga el modelo guardado con Save Model de Orange."""
    return pickle.loads(datos)


def predecir(modelo, valores: dict):
    """
    Devuelve (probabilidad_de_atipico, etiqueta).
    Arma la fila en el mismo orden de columnas que el modelo aprendió.
    """
    import Orange  # se importa aquí para dar un error claro si falta

    # Orange guarda el dominio original (las columnas tal como entraron al
    # modelo) y el dominio interno (después de su preprocesamiento). La fila
    # nueva debe armarse con el ORIGINAL; el modelo la convierte solo.
    base = getattr(modelo, "original_domain", None)
    if base is None:
        base = modelo.domain

    columnas = [v.name for v in base.attributes]
    fila = np.array([[float(valores[c]) for c in columnas]])
    # Mismas 6 variables + la clase (vacía), sin columnas meta
    dominio = Orange.data.Domain(base.attributes, base.class_var)
    tabla = Orange.data.Table.from_numpy(dominio, fila, np.full(1, np.nan))

    _, probs = modelo(tabla, ret=Orange.classification.Model.ValueProbs)
    clases = list(modelo.domain.class_var.values)
    p_atipico = float(probs[0][clases.index("Yes")])
    return p_atipico, ("Atípico" if p_atipico >= 0.5 else "Normal")


def leer_foto(imagen) -> str:
    """OCR con Tesseract. Intenta español y, si no está instalado, usa el idioma por defecto."""
    import pytesseract
    from PIL import Image

    img = Image.open(imagen)
    try:
        return pytesseract.image_to_string(img, lang="spa").strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(img).strip()


# ---------------------------------------------------------------- interfaz
st.title("💊 Detector de lecturas atípicas")
st.caption(
    "Evalúa si la lectura OCR de una medicina es atípica. "
    "Esto **no confirma** que una medicina sea falsa."
)

with st.sidebar:
    st.header("Modelo")
    subido = st.file_uploader("Archivo .pkcls de Orange", type=["pkcls"])
    if subido is not None:
        datos_modelo = subido.getvalue()
    elif RUTA_MODELO.exists():
        datos_modelo = RUTA_MODELO.read_bytes()
        st.success("Usando modelo_ocr.pkcls de la carpeta de la app.")
    else:
        datos_modelo = None
        st.warning("Sube tu archivo .pkcls o ponlo en la carpeta como modelo_ocr.pkcls")

nombre = st.text_input("Nombre del medicamento (como debería aparecer)")

modo = st.radio("Texto del empaque", ["Pegar texto OCR", "Subir foto"], horizontal=True)
texto_ocr = ""
if modo == "Pegar texto OCR":
    texto_ocr = st.text_area("Texto OCR", height=120)
else:
    foto = st.file_uploader("Foto del empaque", type=["png", "jpg", "jpeg"])
    if foto is not None:
        st.image(foto, width=300)
        try:
            texto_ocr = leer_foto(foto)
        except Exception as e:  # Tesseract no instalado, etc.
            st.error(f"No se pudo leer la foto: {e}")
        texto_ocr = st.text_area("Texto leído (puedes corregirlo)", value=texto_ocr, height=120)

if st.button("Evaluar", type="primary"):
    if not nombre.strip() or not texto_ocr.strip():
        st.error("Completa el nombre y el texto OCR.")
        st.stop()
    if datos_modelo is None:
        st.error("Falta el modelo (.pkcls).")
        st.stop()

    valores = calcular_variables(nombre, texto_ocr)

    st.subheader("Variables calculadas")
    st.table({k: [valores[k]] for k in NOMBRES_VARIABLES})

    try:
        modelo = cargar_modelo(datos_modelo)
        p, etiqueta = predecir(modelo, valores)
    except ModuleNotFoundError:
        st.error("Falta instalar Orange3:  pip install Orange3")
        st.stop()
    except Exception as e:
        st.error(f"No se pudo usar el modelo: {e}")
        st.stop()

    st.subheader("Resultado")
    if etiqueta == "Atípico":
        st.warning(f"Lectura **atípica** · probabilidad {p:.0%}")
        st.write("Conviene volver a tomar la foto o verificar el producto en una fuente oficial.")
    else:
        st.success(f"Lectura **normal** · probabilidad de atípico {p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))

with st.expander("Limitaciones"):
    st.markdown(
        "- El modelo se entrenó con **22 atípicos** de 451 filas (Recall 0.61, Precision 0.78).\n"
        "- Aprendió a copiar un Isolation Forest, **no** a distinguir medicinas falsas.\n"
        "- Los datos de entrenamiento son texto de un listado impreso, no fotos de empaques reales.\n"
        "- Para detectar falsificaciones hace falta cotejar registro sanitario y lote con una base oficial."
    )

"""
Cálculo de las 6 variables que usa el modelo de Orange.

Las fórmulas se dedujeron comparando "Nombre del medicamento" y "Texto OCR"
contra los valores de la hoja Datos_Orange (451 filas):

  sim_nombre_ocr        100.0 % de coincidencia
  nombre_en_ocr         100.0 %
  largo_ocr             100.0 %
  ocr_con_simbolos      100.0 %
  ocr_confusion_digito  100.0 %
  tiene_concentracion    99.8 % (450 de 451)
"""
import re
from difflib import SequenceMatcher

NOMBRES_VARIABLES = [
    "sim_nombre_ocr",
    "nombre_en_ocr",
    "largo_ocr",
    "tiene_concentracion",
    "ocr_con_simbolos",
    "ocr_confusion_digito",
]

_UNIDADES = r"(?:mg|mcg|µg|ug|g|kg|ml|mL|l|L|ui|UI|u|U|%|meq|mEq|mmol|mol)"
_RE_CONCENTRACION = re.compile(r"\d+(?:[.,]\d+)?\s*" + _UNIDADES + r"(?![A-Za-z!|])")
_RE_CONFUSION = re.compile(r"\b[iIlL]g\b|[SO]\d[OS]")


def _tiene_concentracion(texto: str) -> int:
    """1 si hay un número seguido de una unidad (ej. 500mg, 25%, 10 mL)."""
    return int(bool(_RE_CONCENTRACION.search(texto)))


def calcular_variables(nombre: str, texto_ocr: str) -> dict:
    """Devuelve las 6 variables como diccionario {nombre_variable: valor}."""
    nombre = (nombre or "").strip()
    texto_ocr = (texto_ocr or "").strip()

    tiene_conc = _tiene_concentracion(texto_ocr)

    return {
        # parecido entre el nombre esperado y todo el texto leído (0 a 1)
        "sim_nombre_ocr": round(
            SequenceMatcher(None, nombre.lower(), texto_ocr.lower()).ratio(), 3
        ),
        # 1 si el nombre aparece tal cual dentro del texto leído
        "nombre_en_ocr": int(nombre.lower() in texto_ocr.lower()),
        # cantidad de caracteres del texto leído
        "largo_ocr": len(texto_ocr),
        "tiene_concentracion": tiene_conc,
        # cantidad de barras "|" (basura típica del OCR)
        "ocr_con_simbolos": texto_ocr.count("|"),
        # letras que parecen dígitos (S5O, ig, lg), solo si no se leyó concentración
        "ocr_confusion_digito": int(
            (not tiene_conc) and bool(_RE_CONFUSION.search(texto_ocr))
        ),
    }

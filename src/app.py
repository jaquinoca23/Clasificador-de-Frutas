"""
app.py — Backend Flask para Clasificador de Frutas en Tiempo Real
"""

import io
import os
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

model = None
MODEL_PATH = "../modelos/clasificador_v1.h5"

CLASES = [
    "Apple Braeburn", "Apple Granny Smith", "Apricot", "Avocado",
    "Banana", "Blueberry", "Cactus fruit", "Cantaloupe",
    "Cherry", "Clementine", "Corn", "Cucumber Ripe",
    "Grape Blue", "Kiwi", "Lemon", "Limes",
    "Mango", "Onion White", "Orange", "Papaya",
    "Passion Fruit", "Peach", "Pear", "Pepper Green",
    "Pepper Red", "Pineapple", "Plum", "Pomegranate",
    "Potato Red", "Raspberry", "Strawberry", "Tomato",
    "Watermelon"
]

EMOJIS = [
    "🍎", "🍏", "🍑", "🥑",
    "🍌", "🫐", "🌵", "🍈",
    "🍒", "🍊", "🌽", "🥒",
    "🍇", "🥝", "🍋", "🍋",
    "🥭", "🧅", "🍊", "🍈",
    "🍈", "🍑", "🍐", "🫑",
    "🌶️", "🍍", "🍑", "🍎",
    "🥔", "🫐", "🍓", "🍅",
    "🍉"
]

IMG_SIZE = (100, 100)
UMBRAL   = 0.99

# Carpeta donde se guardan las imágenes de feedback
FEEDBACK_DIR = "../dataset/feedback"


def load_model():
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model as keras_load
            model = keras_load(MODEL_PATH)
            print(f"✅  Modelo cargado: {MODEL_PATH}")
            print(f"    Input:  {model.input_shape}")
            print(f"    Output: {model.output_shape}  → {len(CLASES)} clases")
        except Exception as exc:
            print(f"⚠️  Error cargando modelo: {exc}")
    return model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_bytes: bytes) -> dict:
    mdl = load_model()
    if mdl is None:
        return {
            "clase": "Modelo no cargado", "emoji": "❌",
            "confianza": 0.0, "porcentaje": "0.0%",
            "indice": -1, "reconocida": False,
        }

    tensor    = preprocess_image(image_bytes)
    preds     = mdl.predict(tensor, verbose=0)[0]
    idx       = int(np.argmax(preds))
    confianza = round(float(preds[idx]), 4)
    reconocida = confianza >= UMBRAL
    nombre = CLASES[idx] if idx < len(CLASES) else f"Clase {idx}"
    emoji  = EMOJIS[idx] if idx < len(EMOJIS) else "🍽️"

    print(f"  DEBUG: {nombre} ({confianza*100:.1f}%) {'✅' if reconocida else '❌'}")

    return {
        "clase":      nombre,
        "emoji":      emoji,
        "confianza":  confianza,
        "porcentaje": f"{confianza * 100:.1f}%",
        "indice":     idx,
        "reconocida": reconocida,
    }


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        if request.content_type and "multipart" in request.content_type:
            image_bytes = request.files["frame"].read()
        elif request.is_json:
            data = request.get_json(force=True)
            b64 = data.get("frame", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        else:
            image_bytes = request.data
            if not image_bytes:
                return jsonify({"error": "Sin datos de imagen"}), 400

        result = predict(image_bytes)
        return jsonify(result)

    except Exception as exc:
        print(f"ERROR en /predict: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/feedback", methods=["POST"])
def feedback_endpoint():
    """
    Recibe el feedback del usuario (👍 correcto / 👎 incorrecto).
    Guarda la imagen en dataset/feedback/<etiqueta_correcta>/
    Body JSON:
      {
        "frame":          "<base64>",
        "prediccion":     "Banana",
        "correcto":       true/false,
        "correccion":     "Kiwi"   (solo si correcto=false)
      }
    """
    try:
        data       = request.get_json(force=True)
        b64        = data.get("frame", "")
        prediccion = data.get("prediccion", "unknown")
        correcto   = data.get("correcto", True)
        correccion = data.get("correccion", prediccion)

        # Etiqueta real = predicción si fue correcto, correccion si no
        etiqueta = prediccion if correcto else correccion

        # Decodificar imagen
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        image_bytes = base64.b64decode(b64)

        # Crear carpeta destino
        carpeta = os.path.join(FEEDBACK_DIR, etiqueta)
        os.makedirs(carpeta, exist_ok=True)

        # Guardar imagen con timestamp
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"fb_{ts}.jpg"
        filepath = os.path.join(carpeta, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        tipo = "✅ CORRECTO" if correcto else f"❌ CORREGIDO → {correccion}"
        print(f"  FEEDBACK {tipo}: guardado en {filepath}")

        return jsonify({
            "ok":      True,
            "archivo": filepath,
            "etiqueta": etiqueta,
        })

    except Exception as exc:
        print(f"ERROR en /feedback: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """Devuelve cuántas imágenes de feedback hay por clase."""
    stats = {}
    if os.path.exists(FEEDBACK_DIR):
        for clase in os.listdir(FEEDBACK_DIR):
            ruta = os.path.join(FEEDBACK_DIR, clase)
            if os.path.isdir(ruta):
                stats[clase] = len(os.listdir(ruta))
    return jsonify({"feedback": stats, "total": sum(stats.values())})


@app.route("/health")
def health():
    mdl = load_model()
    return jsonify({
        "status":         "ok",
        "modelo_cargado": mdl is not None,
        "clases":         CLASES,
        "num_clases":     len(CLASES),
        "umbral":         UMBRAL,
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
"""
app.py — Backend Flask para Clasificador de Frutas en Tiempo Real
"""

import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

model = None
MODEL_PATH = "../modelos/clasificador_completo.h5"

# ✅ 33 clases en el orden EXACTO del clases.txt
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
    "�passion_fruit", "🍑", "🍐", "🫑",
    "🌶️", "🍍", "🍑", "🍎",
    "🥔", "🫐", "🍓", "🍅",
    "🍉"
]

# Emojis corregidos (sin nombres inválidos)
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
        return {"clase": "Modelo no cargado", "emoji": "❌", "confianza": 0.0, "porcentaje": "0.0%", "indice": -1}

    tensor = preprocess_image(image_bytes)
    preds  = mdl.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(preds))
    confianza = round(float(preds[idx]), 4)

    print(f"  DEBUG: {CLASES[idx] if idx < len(CLASES) else idx} ({confianza*100:.1f}%)")

    return {
        "clase":      CLASES[idx] if idx < len(CLASES) else f"Clase {idx}",
        "emoji":      EMOJIS[idx] if idx < len(EMOJIS) else "🍽️",
        "confianza":  confianza,
        "porcentaje": f"{confianza * 100:.1f}%",
        "indice":     idx,
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


@app.route("/health")
def health():
    mdl = load_model()
    return jsonify({
        "status": "ok",
        "modelo_cargado": mdl is not None,
        "clases": CLASES,
        "num_clases": len(CLASES),
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
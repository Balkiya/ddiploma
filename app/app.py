from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
# === Загрузка моделей ===
with open("models/gesture_type_model.pkl", "rb") as f:
    type_model = pickle.load(f)

with open("models/gesture_static_model.pkl", "rb") as f:
    static_model = pickle.load(f)

dynamic_model = load_model("models/lstm_dynamic_model.keras")
# === Метки для динамики ===
DYNAMIC_DATA_DIR = '../scripts/gesture_type_data/dynamic'
dynamic_labels = sorted([
    folder for folder in os.listdir(DYNAMIC_DATA_DIR)
    if os.path.isdir(os.path.join(DYNAMIC_DATA_DIR, folder))
])

SEQUENCE_LENGTH = 30

# === Главная страница ===
@app.route("/")
def index():
    return render_template("index.html")

# === Обучающая страница ===
@app.route("/learn")
def learn():
    try:
        video_dir = "static/videos"
        letters = sorted([
            filename.replace(".mp4", "") for filename in os.listdir(video_dir)
            if filename.endswith(".mp4")
        ])
    except Exception as e:
        letters = []
    return render_template("learn.html", letters=letters)
# === Предсказание жеста ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    points = data.get("data", [])
    gesture_type = data.get("type", "auto")

    if not points:
        return jsonify(prediction="Нет данных", type="Ошибка")
    # === Определение типа жеста ===
    if gesture_type == "auto":
        flat = np.array(points[-1]).flatten()
        if flat.shape[0] == 42:
            gesture_type = type_model.predict([flat])[0]
        else:
            return jsonify(prediction="Ошибка: недопустимый вектор", type="Ошибка")

    # === Статический жест ===
    if gesture_type == "static":
        flat = np.array(points[-1]).flatten()
        if len(flat) != 42:
            return jsonify(prediction="Ошибка: недопустимая длина", type="Ошибка")
        pred = static_model.predict([flat])[0]
        return jsonify(prediction=pred, type="static")

    # === Динамический жест ===
    elif gesture_type == "dynamic":
        if len(points) < SEQUENCE_LENGTH:
            return jsonify(prediction="Ожидание кадров...", type="dynamic")
        sequence = np.array(points[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, -1)
        probs = dynamic_model.predict(sequence)[0]
        pred_index = int(np.argmax(probs))
        label = dynamic_labels[pred_index] if pred_index < len(dynamic_labels) else f"Н/Д_{pred_index}"
        return jsonify(prediction=label, type="dynamic")

    return jsonify(prediction="Неизвестный тип", type="Ошибка")




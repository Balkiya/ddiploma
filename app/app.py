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



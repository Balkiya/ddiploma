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

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
# === Параметры ===
DATA_DIR = './gesture_type_data/dynamic'
MODEL_PATH = './models/mlp_dynamic_model.keras'
SEQUENCE_LENGTH = 30

# === Загрузка данных ===
X, y = [], []
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    files = sorted(os.listdir(label_dir))
    for i in range(len(files) - SEQUENCE_LENGTH):
        sequence = []
        for j in range(SEQUENCE_LENGTH):
            file_path = os.path.join(label_dir, files[i + j])
            keypoints = np.load(file_path)
            sequence.append(keypoints.flatten())
        X.append(sequence)
        y.append(label_map[label])

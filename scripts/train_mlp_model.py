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
X = np.array(X)  # shape: (samples, 30, 42)
y = to_categorical(y)

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# === MLP модель ===
model = Sequential()
model.add(Flatten(input_shape=(SEQUENCE_LENGTH, X.shape[2])))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Обучение модели ===
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
def plot_training_history(history, title_prefix=""):
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Train accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Val accuracy')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Train loss')
    plt.plot(epochs, val_loss, 'ro-', label='Val loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

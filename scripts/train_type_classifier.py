import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
DATA_DIR = './gesture_type_data'
MODEL_PATH = '../app/models/gesture_type_model.pkl'
X, y = [], []

# Рекурсивно ищем .npy в подпапках
for gesture_type in os.listdir(DATA_DIR):
    gesture_path = os.path.join(DATA_DIR, gesture_type)

    if not os.path.isdir(gesture_path):
        continue

    for letter in os.listdir(gesture_path):
        letter_path = os.path.join(gesture_path, letter)

        if not os.path.isdir(letter_path):
            continue

        for file in os.listdir(letter_path):
            if file.endswith('.npy'):
                path = os.path.join(letter_path, file)
                keypoints = np.load(path).flatten()

                # Проверяем, что размер совпадает (например, 42 * 2 = 84)
                if keypoints.shape[0] not in [42 * 2, 21 * 2]:  # допустимые размеры
                    continue

                X.append(keypoints)
                y.append(gesture_type)  # Сохраняем именно "static" или "dynamic"

X = np.array(X)
y = np.array(y)
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Обучение модели
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
model.fit(X_train, y_train)
# Оценка
acc = model.score(X_test, y_test)
print(f"Точность определителя типа жеста (static/dynamic): {acc:.2f}")


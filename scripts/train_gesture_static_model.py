import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Папка с данными для статических жестов
DATA_DIR = './gesture_type_data/static'
MODEL_PATH = '../app/models/gesture_static_model.pkl'

X, y = [], []
# Загрузка данных
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file_name in os.listdir(label_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(label_path, file_name)
            keypoints = np.load(file_path).flatten()
            X.append(keypoints)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"🎯 Точность новой статической модели: {accuracy:.2f}")

# Сохранение модели
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Модель gesture_static_model.pkl сохранена!")

import cv2
import numpy as np
import os
import pickle
from collections import deque
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model

# === Пути к моделям и папкам ===
STATIC_MODEL_PATH = '../app/models/gesture_static_model.pkl'
DYNAMIC_MODEL_PATH = '../app/models/transformer_dynamic_model.keras'
TYPE_MODEL_PATH = '../app/models/gesture_type_model.pkl'

STATIC_DATA_DIR = './gesture_type_data/static'
DYNAMIC_DATA_DIR = './gesture_type_data/dynamic'

# === Загрузка моделей ===
with open(STATIC_MODEL_PATH, 'rb') as f:
    static_model = pickle.load(f)

dynamic_model = load_model(DYNAMIC_MODEL_PATH)

with open(TYPE_MODEL_PATH, 'rb') as f:
    type_model = pickle.load(f)

# === Загрузка меток ===
static_labels = sorted([folder for folder in os.listdir(STATIC_DATA_DIR) if os.path.isdir(os.path.join(STATIC_DATA_DIR, folder))])
dynamic_labels = sorted([folder for folder in os.listdir(DYNAMIC_DATA_DIR) if os.path.isdir(os.path.join(DYNAMIC_DATA_DIR, folder))])

# === MediaPipe ===
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# === Шрифт ===
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 32)

# === Буфер для динамических жестов ===
SEQUENCE_LENGTH = 30
dynamic_sequence = deque(maxlen=SEQUENCE_LENGTH)

# === Камера ===
cap = cv2.VideoCapture(0)
print("📸 Камера запущена. Показывай жест...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    prediction_text = "Жест не распознан"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            keypoints_np = np.array(keypoints).reshape(1, -1)

            # Определяем тип жеста
            gesture_type = type_model.predict(keypoints_np)[0]

            if gesture_type == "static":
                if keypoints_np.shape[1] == static_model.n_features_in_:
                    prediction = static_model.predict(keypoints_np)[0]
                    prediction_text = f"Статическая буква: {prediction}"
                else:
                    prediction_text = "Ошибка признаков (статическая)"
            else:
                dynamic_sequence.append(keypoints)

                if len(dynamic_sequence) == SEQUENCE_LENGTH:
                    input_sequence = np.array(dynamic_sequence).reshape(1, SEQUENCE_LENGTH, -1)
                    dynamic_pred = dynamic_model.predict(input_sequence)[0]
                    predicted_index = np.argmax(dynamic_pred)
                    prediction_text = f"Динамическая буква: {dynamic_labels[predicted_index]}"

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 10), prediction_text, font=font, fill=(0, 255, 0))
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Комбинированное распознавание жестов", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

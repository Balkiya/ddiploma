import cv2
import numpy as np
import os
import mediapipe as mp

SAVE_DIR = "./gesture_type_data"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("📸 Камера запущена.")

# Выбор типа (static / dynamic)
gesture_type = input("👉 Введите тип жеста (static / dynamic): ").strip().lower()
if gesture_type not in ["static", "dynamic"]:
    print("❌ Неверный тип жеста. Введите static или dynamic.")
    exit()

# Ввод буквы
letter = input("👉 Введите букву (пример: A, Б, CH и т.д.): ").strip().upper()

# Папка сохранения
save_path = os.path.join(SAVE_DIR, gesture_type, letter)
os.makedirs(save_path, exist_ok=True)

count = 0
NUM_SAMPLES = 200

print(f"🤚 Показывай жест для '{letter}' ({gesture_type}). Сохраняем {NUM_SAMPLES} примеров...")

while count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            keypoints = np.array(keypoints)
            np.save(os.path.join(save_path, f"{count}.npy"), keypoints)
            count += 1
            print(f"[{count}/{NUM_SAMPLES}] сохранено.")

    cv2.putText(frame, f"{gesture_type.upper()} {letter} ({count}/{NUM_SAMPLES})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Сбор данных для определителя", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Сбор завершен.")

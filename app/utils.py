import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_image(image_b64):
    import base64
    import io
    from PIL import Image

    image_data = base64.b64decode(image_b64.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    drawn = image_bgr.copy()
    keypoints = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(drawn, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

    _, buffer = cv2.imencode(".jpg", drawn)
    drawn_b64 = base64.b64encode(buffer).decode("utf-8")
    return keypoints, f"data:image/jpeg;base64,{drawn_b64}"

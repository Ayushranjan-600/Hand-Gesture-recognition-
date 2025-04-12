
import cv2
import mediapipe as mp
import numpy as np
import joblib
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import pyautogui
import keyboard
import time

# Load model and tools
clf = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

# Volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Screen size
screen_width, screen_height = pyautogui.size()

# Cooldown setup
last_action_time = 0
cooldown = 1.5  # seconds


def change_brightness(action):
    current_brightness = sbc.get_brightness(display=0)[0]
    if action == "increase_brightness":
        new_brightness = min(100, current_brightness + 10)
    elif action == "decrease_brightness":
        new_brightness = max(0, current_brightness - 10)
    sbc.set_brightness(new_brightness, display=0)


def change_volume(action):
    current_volume = volume.GetMasterVolumeLevelScalar()
    if action == "increase_vol":
        new_volume = min(1.0, current_volume + 0.1)
    elif action == "decrease_vol":
        new_volume = max(0.0, current_volume - 0.1)
    volume.SetMasterVolumeLevelScalar(new_volume, None)


def play_pause_media():
    keyboard.press_and_release('space')


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)

            gesture_index = clf.predict(landmarks)[0]
            gesture_name = label_encoder.inverse_transform([gesture_index])[0]

            cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()
            if gesture_name in ["increase_brightness", "decrease_brightness"]:
                if current_time - last_action_time > cooldown:
                    change_brightness(gesture_name)
                    last_action_time = current_time

            elif gesture_name in ["increase_vol", "decrease_vol"]:
                if current_time - last_action_time > cooldown:
                    change_volume(gesture_name)
                    last_action_time = current_time

            elif gesture_name == "play_pause":
                if current_time - last_action_time > cooldown:
                    play_pause_media()
                    last_action_time = current_time

           

    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

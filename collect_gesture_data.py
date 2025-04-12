import cv2
import mediapipe as mp
import pandas as pd
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Store gesture data
data = []
labels = []

# Define gestures and their assigned keys
GESTURE_KEY_MAP = {
    '1': 'increase_vol',
    '2': 'decrease_vol',
    '3': 'increase_brightness',
    '4': 'decrease_brightness',
    '5': 'play_pause'
}

print("📸 Press a number key to label the gesture:")
for k, v in GESTURE_KEY_MAP.items():
    print(f"  [{k}] -> {v}")

# Timer to control data sampling rate
last_saved_time = time.time()
cooldown = 2.0  # seconds between samples

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                current_time = time.time()
                key = cv2.waitKey(1) & 0xFF

                # Check if key corresponds to a gesture and cooldown passed
                if chr(key) in GESTURE_KEY_MAP and current_time - last_saved_time > cooldown:
                    gesture_label = GESTURE_KEY_MAP[chr(key)]
                    data.append(landmarks)
                    labels.append(gesture_label)
                    last_saved_time = current_time
                    print(f"✅ Saved sample for gesture: {gesture_label}")

        # Show frame
        cv2.imshow("Gesture Data Collection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save dataset
df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("gesture_data.csv", index=False)

print("✅ Data saved to 'gesture_data.csv'")



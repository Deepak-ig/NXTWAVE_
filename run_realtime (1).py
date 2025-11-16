import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import pyttsx3
import time

# Load trained model
model = load("models/gesture_model.joblib")

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------- Parameters ----------
COOLDOWN = 3.0         # seconds between accepted letters
MIN_HOLD = 0.0         # seconds label must be held to be "stable"
PROB_THRESHOLD = 0.60  # min confidence to consider prediction valid
SPEAK_DELAY = 1.0      # seconds without hand to trigger speaking
# -------------------------------

# ---------- State ----------
buffered_word = ""
last_append_time = 0.0
last_label = None
last_detect_time = time.time()

prev_frame_label = None
label_start_time = None
# ---------------------------

print("Press ESC to quit")

def extract_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    coords -= coords[0]  # normalize to wrist
    return coords.flatten()

def speak(text):
    """Speak text safely (fresh engine each time)."""
    eng = pyttsx3.init()
    eng.say(text)
    eng.runAndWait()
    eng.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    stable_label = None
    frame_pred_label = None
    pred_conf = 0.0

    if results.multi_hand_landmarks:
        # Use the first detected hand
        hl = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # Predict
        features = extract_landmarks(hl).reshape(1, -1)
        try:
            frame_pred_label = model.predict(features)[0]
        except Exception:
            frame_pred_label = None

        # Try to get confidence
        try:
            probs = model.predict_proba(features)[0]
            pred_conf = float(np.max(probs))
        except Exception:
            pred_conf = 1.0

        if frame_pred_label is not None and pred_conf >= PROB_THRESHOLD:
            # update hold timer
            if frame_pred_label != prev_frame_label:
                prev_frame_label = frame_pred_label
                label_start_time = current_time
            if label_start_time is not None and (current_time - label_start_time) >= MIN_HOLD:
                stable_label = frame_pred_label
        else:
            prev_frame_label = None
            label_start_time = None
            stable_label = None

        # show predicted/confidence on-screen
        cv2.putText(frame, f"Pred: {str(frame_pred_label)} {pred_conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if stable_label:
            cv2.putText(frame, f"Stable: {stable_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # Append logic: only if cooldown passed AND stable_label changed
            if stable_label.lower() == "space":
                if last_label != "space" and (current_time - last_append_time >= COOLDOWN):
                    buffered_word += " "
                    last_append_time = current_time
                    last_label = "space"
            elif stable_label.lower() != "closed_hand":
                if stable_label != last_label and (current_time - last_append_time >= COOLDOWN):
                    buffered_word += stable_label
                    last_append_time = current_time
                    last_label = stable_label

        last_detect_time = current_time  # hand present

    else:
        # No hand detected
        if buffered_word and (current_time - last_detect_time > SPEAK_DELAY):
            print(f"Speaking: {buffered_word}")
            speak(buffered_word)
            # reset state so next word works cleanly
            buffered_word = ""
            last_label = None
            last_append_time = 0.0
            prev_frame_label = None
            label_start_time = None

    # show the building word on screen
    cv2.putText(frame, f"Word: {buffered_word}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Hand Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

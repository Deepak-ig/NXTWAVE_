import cv2
import mediapipe as mp
import numpy as np
import csv, os

# Make sure data folder exists
os.makedirs("data", exist_ok=True)
csv_path = "data/landmarks.csv"

# If file doesn't exist, create header row
if not os.path.exists(csv_path):
    header = []
    for i in range(21):  # 21 landmarks
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_label = None  # will be set when key pressed

def extract_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    coords -= coords[0]  # normalize to wrist
    return coords.flatten().tolist()

print("=== Controls ===")
print("A–Z : set letter label")
print("0–9 : set number label")
print("Spacebar : set label to 'space'")
print("C : set label to 'closed_hand'")
print("S : save a sample for the current label")
print("ESC : quit")
print("================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if ord('A') <= key <= ord('Z'):  # A-Z
        current_label = chr(key)
        print(f"Current label set to {current_label}")
    elif ord('0') <= key <= ord('9'):  # 0-9
        current_label = chr(key)
        print(f"Current label set to {current_label}")
    elif key == 32:  # Spacebar
        current_label = "space"
        print("Current label set to space")
    elif key == ord('c'):  # closed_hand
        current_label = "closed_hand"
        print("Current label set to closed_hand")
    elif key == 27:  # ESC to quit
        break

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if key == ord('s') and current_label:
                row = extract_landmarks(hl)
                row.append(current_label)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"Saved sample for {current_label}")

    # Show label info
    label_text = f"Current: {current_label}" if current_label else "Press A–Z/0–9/Space"
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Data", frame)

cap.release()
cv2.destroyAllWindows()

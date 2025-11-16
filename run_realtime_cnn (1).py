# run_realtime_cnn.py
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/asl_model.h5")

# Get class names (must match training order)
class_names = ['A','B','C', ...]  # fill in exactly as printed by train_cnn.py

img_height, img_width = 64, 64

cap = cv2.VideoCapture(0)

current_word = ""

while True:
    ret, frame = cap.read()
    if not ret: break
    # center crop / resize frame to model input
    img = cv2.resize(frame, (img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1,64,64,3)
    img_array = img_array / 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    cv2.putText(frame, f"{predicted_class} {confidence:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("ASL Realtime", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

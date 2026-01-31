import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# =========================
# PATHS (DO NOT HARD-CODE ABSOLUTE PATHS)
# =========================
MODEL_PATH = "../models/emotion_cnn_fer2013.keras"
FACE_CASCADE_PATH = "../face_detection/haarcascade_frontalface_default.xml"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels (MUST match training order)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# =========================
# SMOOTHING (IMPORTANT FOR REAL TIME)
# =========================
prediction_queue = deque(maxlen=5)

# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # =========================
        # PREPROCESS (MUST MATCH TRAINING)
        # =========================
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # =========================
        # PREDICTION
        # =========================
        preds = model.predict(face, verbose=0)[0]
        prediction_queue.append(preds)

        avg_preds = np.mean(prediction_queue, axis=0)
        emotion = EMOTIONS[np.argmax(avg_preds)]
        confidence = np.max(avg_preds)

        # =========================
        # DISPLAY
        # =========================
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
